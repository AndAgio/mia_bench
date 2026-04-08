from typing import Union
import torch
from torch.utils.data import DataLoader
from src.data.helpers import SubsampledDataset
from src.utils.configs import DefenderConfigs, PurifierDefenseConfigs, TrainConfigs
from src.mia.defenses.base import BaseDefender

BATCH_SIZE = 64

class PurifierDefender(BaseDefender):
    # Implementation of Purifier defense from "Purifier: Defending Data Inference Attacks via Transforming Confidence Scores" (https://ojs.aaai.org/index.php/AAAI/article/view/26289).
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, PurifierDefenseConfigs), f"PurifierDefender can only be used with PurifierDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'purifier_defender'
        self.purifier_configs = defender_configs.defense
        self.reformer = ConfidenceReformer(num_classes=self.model_configs.num_classes,
                                        latent_dim=self.purifier_configs.reformer_latent_dim,
                                        hidden=self.purifier_configs.reformer_hidden_dim)
        self.label_swapper = TorchLabelSwapper()

    def train_model(self, train_configs, return_stats: bool = False):
        self.logger.print_it("PurifierDefender: training original model...")
        return super().train_model(train_configs=train_configs, return_stats=return_stats)

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        self.device = self.get_device(device)
        self.logger.print_it("PurifierDefender: starting training of purifier module...")
        self.train_reformer()
        self.logger.print_it("PurifierDefender: finished training reformer, building Pindex for label swapping...")
        Pindex, Pindex_labels = self.build_pindex()
        self.label_swapper.set_attributes(Pindex, Pindex_labels)
        self.label_swapper.set_device(self.device)
        self.logger.print_it('PurifierDefender: building defended model to be used for future prediction...')
        self.defended_model = PurifiedModelWrapper(original_model=self.trained_model,
                                                reformer=self.reformer,
                                                label_swapper=self.label_swapper,
                                                threshold=self.purifier_configs.swap_threshold).to(self.device)
        self.logger.print_it('PurifierDefender: finished building defended torch module!')
        return self.defended_model

    def train_original_model(self, train_configs: TrainConfigs):
        self.logger.print_it("PurifierDefender: training original model...")
        return super().train_model(train_configs=train_configs, return_stats=False)

    def train_reformer(self):
        try:
            reformer_training_dataset = self.dataset.get('val')
        except (KeyError, AttributeError):
            raise ValueError("PurifierDefender: no validation split found in dataset! Purifier reformer requires a reference dataset to be trained!")
        self.logger.print_it("PurifierDefender: training reformer model on the validation dataset...")
        dataloader = DataLoader(reformer_training_dataset, batch_size=self.purifier_configs.reformer_batch_size, shuffle=True)
        
        self.original_model = self.trained_model.to(self.device).eval()
        self.reformer.to(self.device).train()

        optimizer = torch.optim.Adam(self.reformer.parameters(), lr=self.purifier_configs.reformer_lr)

        for ep in range(self.purifier_configs.reformer_epochs):
            cum_loss = 0.0
            cum_acc = 0.0
            for (inputs, _, _, _) in dataloader:
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    c = torch.nn.functional.softmax(self.original_model(inputs), dim=-1)
                    l = torch.nn.functional.one_hot(c.argmax(dim=-1), num_classes=c.size(1)).float()
                c_hat = self.reformer(c, l)
                loss_R = ((c_hat - c)**2).mean()
                eps = 1e-12
                loss_L = -(l * (c_hat + eps).log()).sum(dim=-1).mean()
                loss = loss_R + self.purifier_configs.reformer_lambda * loss_L
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
                reformer_acc = (c_hat.argmax(dim=-1) == l.argmax(dim=-1)).float().mean().item()
                cum_acc += reformer_acc
            cum_loss /= len(dataloader)
            cum_acc /= len(dataloader)
            self.logger.print_it_same_line(f"PurifierDefender: training reformer model | [Epoch {ep+1}/{self.purifier_configs.reformer_epochs}] | Loss = {cum_loss:.4f}, Reformer Acc = {cum_acc:.4f}...", console_only=True)
        self.logger.set_logger_newline(console_only=True)
        self.logger.print_it("PurifierDefender: finished training reformer model!")

    def build_pindex(self):
        Pindex = []
        Pindex_labels = []
        p_dataset_indices = self.dataset.get('train').get_indices(mode='original')[:self.purifier_configs.pindex_size]
        p_dataset = SubsampledDataset(self.dataset.get('train'), original_indices=p_dataset_indices, strict=True)
        # p_dataset = Subset(self.dataset.get('train'), indices=range(self.purifier_configs.pindex_size))
        dataloader = DataLoader(p_dataset, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for (inputs, labels, _, _) in dataloader:
                inputs = inputs.to(self.device)
                c = torch.nn.functional.softmax(self.trained_model(inputs), dim=-1)
                batch_size = inputs.size(0)
                for i in range(batch_size):
                    Pindex.append(c[i].cpu())
                    Pindex_labels.append(int(labels[i]))
        return Pindex, Pindex_labels



class ConfidenceReformer(torch.nn.Module):
    """
    PURIFIER confidence reformer G_theta.
    """
    def __init__(self, num_classes, latent_dim=16, hidden=128):
        super().__init__()
        in_dim = num_classes * 2  # c || onehot(label)

        # Encoder
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(hidden, latent_dim)
        self.logvar = torch.nn.Linear(hidden, latent_dim)

        # Decoder
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + num_classes, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, num_classes),
        )
        self.sm = torch.nn.Softmax(dim=-1)
    
    def forward(self, c, l_onehot):
        h = self.enc(torch.cat([c, l_onehot], dim=-1))
        mu = self.mu(h)
        logvar = self.logvar(h)

        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std

        dec_in = torch.cat([z, l_onehot], dim=-1)
        return self.sm(self.dec(dec_in))



class TorchLabelSwapper(torch.nn.Module):
    """
    Pure PyTorch kNN label-swapper.
    Computes nearest neighbors fully on GPU.
    """
    def __init__(self, Pindex: Union[torch.Tensor, list] = None, Pindex_labels: Union[torch.Tensor, list] = None, ):
        """
        Pindex: list of (K,) tensors OR a (T,K) tensor
        Pindex_labels: list/array of T integers
        """
        super().__init__()
        if isinstance(Pindex, list):
            self.P = torch.stack(Pindex)            # (T, K)
        else:
            self.P = Pindex
        
        self.P = self.P.float() if Pindex is not None else None
        self.labels = torch.tensor(Pindex_labels).long() if Pindex_labels is not None else None

    def set_Pindex(self, Pindex: Union[torch.Tensor, list]):
        if isinstance(Pindex, list):
            self.P = torch.stack(Pindex)            # (T, K)
        else:
            self.P = Pindex
        self.P = self.P.float()
    
    def set_Pindex_labels(self, Pindex_labels: Union[torch.Tensor, list]):
        assert Pindex_labels is not None, "Pindex_labels cannot be None when setting Pindex labels!"
        self.labels = torch.tensor(Pindex_labels).long()

    def set_attributes(self, Pindex: Union[torch.Tensor, list], Pindex_labels: Union[torch.Tensor, list]):
        self.set_Pindex(Pindex)
        self.set_Pindex_labels(Pindex_labels)

    def set_device(self, device: torch.device):
        self.device = device
        if self.P is not None:
            self.P = self.P.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)

    @torch.no_grad()
    def query(self, C_batch, threshold=0.1):
        """
        C_batch: (B, K) confidence vectors
        Returns: LongTensor(B,) with label to swap to,
                or -1 for "no swap"
        """
        assert self.P is not None, "TorchLabelSwapper: Pindex must be provided to use the query function!"
        assert self.labels is not None, "TorchLabelSwapper: Pindex_labels must be provided to use the query function!"

        # Ensure on GPU
        C_batch = C_batch.float().to(self.device)  # (B, K)

        # Compute pairwise L2 distances:
        # dist[i, j] = ||c_i - P_j||^2
        # Efficient GPU formulation:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        B, K = C_batch.shape
        T = self.P.shape[0]

        c_norm = (C_batch ** 2).sum(dim=1).view(B, 1)        # (B,1)
        p_norm = (self.P ** 2).sum(dim=1).view(1, T)         # (1,T)
        cross = C_batch @ self.P.t()                          # (B,T)

        dist = c_norm + p_norm - 2 * cross                    # (B,T)

        # 1-NN
        min_dist, min_idx = torch.min(dist, dim=1)            # (B,), (B,)

        # Apply threshold
        swap_labels = torch.where(
            min_dist <= threshold,
            self.labels[min_idx],
            torch.tensor(-1, device=self.device)
        )

        return swap_labels

    def forward(self, inputs, threshold=0.1):
        if self.training:
            raise RuntimeError("TorchLabelSwapper: forward should not be used in training mode!")
        return self.query(inputs, threshold=threshold)


class PurifiedModelWrapper(torch.nn.Module):
    """
    Drop-in model wrapper for MI benchmarking.
    """
    def __init__(self, original_model: torch.nn.Module, reformer: ConfidenceReformer, label_swapper: TorchLabelSwapper, threshold: float=0.1):
        super().__init__()
        self.original_model = original_model
        self.reformer = reformer
        self.label_swapper = label_swapper
        self.threshold = threshold

    @torch.no_grad()
    def defend(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Batched PURIFIER inference: F -> G -> H
        x: (B, ...)
        returns purified confidence vectors p_out (B,K)
        """

        x = inputs
        self.original_model.eval()
        self.reformer.eval()

        # 1. classifier confidences
        logits = self.original_model(x)
        c = torch.nn.functional.softmax(logits, dim=-1)            # (B,K)
        K = c.size(1)

        # 2. condition (predicted labels)
        l = torch.nn.functional.one_hot(
            c.argmax(dim=-1), num_classes=K
        ).float()

        # 3. purified scores
        p = self.reformer(c, l)                     # (B,K)

        # 4. compute swap labels for batch
        swap_labels = self.label_swapper.query(c, threshold=self.threshold)  # (B,)

        # 5. apply swaps
        mask = swap_labels != -1
        if mask.any():
            p[mask] = 0
            p[mask, swap_labels[mask]] = 1.0     # forced one-hot

        return p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("PurifierDefender: PurifiedModelWrapper should not be used in training mode!")
        return self.defend(x)
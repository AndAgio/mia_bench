from typing import Union
import torch
from torch.utils.data import DataLoader
from src.utils.configs import DefenderConfigs, PurifierDefenseConfigs, TrainConfigs
from src.mia.defenses.base import BaseDefender, DefendedModel

BATCH_SIZE = 64

class PurifierDefender(BaseDefender):
    """Purifier: reforms each output with a noisy autoencoder trained on non-members, and swaps some members' labels.

    Paper: "Purifier: Defending Data Inference Attacks via Transforming Confidence Scores",
    https://arxiv.org/abs/2212.00612. Documentation: docs/defenses/purifier.md.
    """
    inference_only = True

    def __init__(self, defender_configs: DefenderConfigs):
        """Check that the configs select Purifier, and build the reformer and the label swapper."""
        assert isinstance(defender_configs.defense, PurifierDefenseConfigs), f"PurifierDefender can only be used with PurifierDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'purifier_defender'
        self.purifier_configs = defender_configs.defense
        self.reformer = ConfidenceReformer(num_classes=self.model_configs.num_classes,
                                        latent_dim=self.purifier_configs.reformer_latent_dim,
                                        hidden=self.purifier_configs.reformer_hidden_dim,
                                        noise_sigma=self.purifier_configs.reformer_noise_sigma)
        self.label_swapper = TorchLabelSwapper()

    def train_model(self, train_configs, return_stats: bool = False):
        """Train normally, since Purifier only changes how the trained model answers queries."""
        self.logger.print_it("PurifierDefender: training original model...")
        return super().train_model(train_configs=train_configs, return_stats=return_stats)

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        """Train the reformer, pick the swap set, and wrap the trained model with them."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.device = device
        self.logger.print_it("PurifierDefender: starting training of purifier module...")
        self.train_reformer()
        self.logger.print_it("PurifierDefender: finished training reformer, building Pindex for label swapping...")
        self.label_swapper.set_Pindex(self.build_pindex())
        self.label_swapper.set_device(self.device)
        self.logger.print_it('PurifierDefender: building defended model to be used for future prediction...')
        self.defended_model = PurifiedModelWrapper(model=self.trained_model,
                                                reformer=self.reformer,
                                                label_swapper=self.label_swapper,
                                                threshold=self.purifier_configs.swap_threshold,
                                                noise_quantization=self.purifier_configs.noise_quantization).to(self.device)
        self.logger.print_it('PurifierDefender: finished building defended torch module!')
        return self.defended_model

    def train_original_model(self, train_configs: TrainConfigs):
        """Train the classifier normally and return it, without the training stats."""
        self.logger.print_it("PurifierDefender: training original model...")
        return super().train_model(train_configs=train_configs, return_stats=False)

    def train_reformer(self):
        """Train the reformer with Eq. 1 on the classifier's confidences on the val (reference) split."""
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
                if inputs.size(0) < 2:
                    continue  # the reformer's batch-normalised latent cannot train on a single sample
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    c = torch.nn.functional.softmax(self.original_model(inputs), dim=-1)
                    l = torch.nn.functional.one_hot(c.argmax(dim=-1), num_classes=c.size(1)).float()
                c_hat = self.reformer(c, l)  # fresh noise in training; inference seeds it per query
                # Eq. 1 with per-sample squared L2; docs/defenses/purifier.md explains the default lambda
                loss_R = ((c_hat - c)**2).sum(dim=-1).mean()
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
        """Pick the members whose labels are swapped, and return their log-confidences."""
        # Paper: swap the labels of a random share p_swap = (acc_train - acc_test) / acc_train of the members,
        # which brings their accuracy down to the test accuracy. The test split holds the auditing
        # non-members, so the test accuracy is estimated on the reference (val) split instead.
        train_log_c, train_correct, train_indices = self.get_log_confidences(self.dataset.get('train'))
        _, val_correct, _ = self.get_log_confidences(self.dataset.get('val'))
        acc_train = train_correct.float().mean().item()
        acc_test = val_correct.float().mean().item()
        p_swap = max(0.0, (acc_train - acc_test) / acc_train) if acc_train > 0 else 0.0
        generator = torch.Generator().manual_seed(self.dataset_configs.seed)
        positions = torch.randperm(len(train_correct), generator=generator)[:round(p_swap * len(train_correct))]
        self.swap_indices = train_indices[positions].tolist()
        self.logger.print_it(f"PurifierDefender: train acc = {acc_train:.4f}, val acc = {acc_test:.4f}, swapping labels of {len(positions)} members (p_swap = {p_swap:.4f})")
        return train_log_c[positions]

    @torch.no_grad()
    def get_log_confidences(self, dataset):
        """Return the classifier's log-confidences on a dataset, whether each prediction is right, and the indices."""
        self.trained_model.to(self.device).eval()
        log_c, correct, indices = [], [], []
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        for (inputs, labels, original_indices, _) in dataloader:
            logits = self.trained_model(inputs.to(self.device))
            log_c.append(torch.nn.functional.log_softmax(logits, dim=-1).cpu())
            correct.append(logits.argmax(dim=-1).cpu() == labels)
            indices.append(original_indices)
        return torch.cat(log_c), torch.cat(correct), torch.cat(indices)



class ConfidenceReformer(torch.nn.Module):
    """Purifier's confidence reformer G: an autoencoder conditioned on the predicted label, with a noisy latent."""
    def __init__(self, num_classes, latent_dim=2, hidden=32, noise_sigma=1.0):
        """Build the appendix's layers [2K, h, 2h, 4h, latent, 4h, 2h, h, K] for K = num_classes and h = hidden."""
        super().__init__()
        in_dim = num_classes * 2  # c || onehot(label)
        self.latent_dim = latent_dim
        self.noise_sigma = noise_sigma

        def hidden_layer(n_in, n_out):
            return [torch.nn.Linear(n_in, n_out), torch.nn.BatchNorm1d(n_out), torch.nn.ReLU()]

        # Encoder: 2K -> h -> 2h -> 4h -> latent
        self.enc = torch.nn.Sequential(
            *hidden_layer(in_dim, hidden),
            *hidden_layer(hidden, 2 * hidden),
            *hidden_layer(2 * hidden, 4 * hidden),
            torch.nn.Linear(4 * hidden, latent_dim),
            # No learnable scale: with a learned or unnormalised scale, training shrinks the latent noise away.
            torch.nn.BatchNorm1d(latent_dim, affine=False),
        )

        # Decoder: latent || onehot(label) -> 4h -> 2h -> h -> K
        self.dec = torch.nn.Sequential(
            *hidden_layer(latent_dim + num_classes, 4 * hidden),
            *hidden_layer(4 * hidden, 2 * hidden),
            *hidden_layer(2 * hidden, hidden),
            # K scores, as the appendix has for Purchase100 and UTKFace; its CIFAR10 and FaceScrub530 sizes are 2K.
            torch.nn.Linear(hidden, num_classes),
        )
        self.sm = torch.nn.Softmax(dim=-1)

    def forward(self, c, l_onehot, noise=None):
        """Return the reformed confidences of c given the one-hot labels.

        noise: optional (B, latent_dim) standard Gaussian draws; fresh draws are used when None.
        """
        r = self.enc(torch.cat([c, l_onehot], dim=-1))
        if noise is None:
            noise = torch.randn_like(r)
        z = r + self.noise_sigma * noise

        dec_in = torch.cat([z, l_onehot], dim=-1)
        return self.sm(self.dec(dec_in))



class TorchLabelSwapper(torch.nn.Module):
    """Purifier's label swapper H: finds the queries that match a swap-set member, by 1-NN on log-confidences."""
    def __init__(self, Pindex: Union[torch.Tensor, list] = None):
        """Keep the swap set's log-confidences, if given: a list of (K,) tensors or a (T, K) tensor."""
        super().__init__()
        self.P = None
        if Pindex is not None:
            self.set_Pindex(Pindex)

    def set_Pindex(self, Pindex: Union[torch.Tensor, list]):
        """Set the swap set's log-confidences: a list of (K,) tensors or a (T, K) tensor."""
        if isinstance(Pindex, list):
            Pindex = torch.stack(Pindex)            # (T, K)
        self.P = Pindex.float()

    def set_device(self, device: torch.device):
        """Move the swap set to the device the queries come on."""
        self.device = device
        if self.P is not None:
            self.P = self.P.to(device)

    @torch.no_grad()
    def query(self, log_C_batch, threshold):
        """Return a (B,) boolean tensor, True where a row of the (B, K) log-confidences matches a swap-set member."""
        assert self.P is not None, "TorchLabelSwapper: Pindex must be provided to use the query function!"

        log_C_batch = log_C_batch.float().to(self.device)  # (B, K)
        if self.P.size(0) == 0:
            return torch.zeros(log_C_batch.size(0), dtype=torch.bool, device=self.device)

        # 1-NN on exact L2 distances: the ||a||^2 + ||b||^2 - 2 a·b expansion loses far more
        # precision than the threshold on log-confidences of large magnitude
        dist = torch.cdist(log_C_batch, self.P, compute_mode='donot_use_mm_for_euclid_dist')  # (B,T)
        return dist.min(dim=1).values <= threshold

    def forward(self, inputs, threshold):
        """Return the matches of query; the swapper has no training mode."""
        if self.training:
            raise RuntimeError("TorchLabelSwapper: forward should not be used in training mode!")
        return self.query(inputs, threshold=threshold)


class PurifiedModelWrapper(DefendedModel):
    """Purifier's defended model: purifies each output with the reformer and swaps the swap-set members' labels."""
    def __init__(self, model: torch.nn.Module, reformer: ConfidenceReformer, label_swapper: TorchLabelSwapper, threshold: float, noise_quantization: float):
        """Wrap the classifier with the trained reformer and the label swapper."""
        super().__init__(model)
        self.reformer = reformer
        self.label_swapper = label_swapper
        self.threshold = threshold
        self.noise_quantization = noise_quantization

    def query_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Return Gaussian latent noise seeded by each query's hash, so a repeated query gets the same output.

        See docs/defenses/purifier.md. x: (B, ...) model inputs; returns (B, latent_dim) noise on x's device.
        """
        noise = []
        for digest in self.query_digests(x, self.noise_quantization):
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'big'))
            noise.append(torch.randn(self.reformer.latent_dim, generator=generator))
        return torch.stack(noise).to(x.device)

    @torch.no_grad()
    def defended_probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the purified (B, K) probabilities of a batch: the classifier F, then the reformer G, then H."""

        x = inputs
        self.model.eval()
        self.reformer.eval()

        # 1. classifier confidences
        logits = self.model(x)
        c = torch.nn.functional.softmax(logits, dim=-1)            # (B,K)
        K = c.size(1)

        # 2. condition (predicted labels)
        l = torch.nn.functional.one_hot(
            c.argmax(dim=-1), num_classes=K
        ).float()

        # 3. purified scores, with the latent noise seeded per query so repeated queries match
        p = self.reformer(c, l, noise=self.query_noise(x))  # (B,K)

        # 4. find swap-set members. Matched on log-confidences: softmax saturates on
        # confident predictions, so distinct inputs can share the same confidence vector
        mask = self.label_swapper.query(torch.nn.functional.log_softmax(logits, dim=-1), threshold=self.threshold)  # (B,)

        # 5. swap their label to the second largest by exchanging the top two scores of p,
        # so p keeps its scores (no one-hot, no exact zeros added)
        if mask.any():
            top, second = p[mask].topk(2, dim=-1).indices.unbind(dim=-1)
            top_scores = p[mask, top]
            p[mask, top] = p[mask, second]
            p[mask, second] = top_scores

        return p

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        """Return the purified probabilities as logits: softmax gives them back exactly."""
        return self.probabilities_to_logits(self.defended_probabilities(x))
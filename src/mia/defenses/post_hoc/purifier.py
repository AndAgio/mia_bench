import hashlib
from typing import Union
import torch
from torch.utils.data import DataLoader
from src.utils.configs import DefenderConfigs, PurifierDefenseConfigs, TrainConfigs
from src.mia.defenses.base import BaseDefender

BATCH_SIZE = 64

class PurifierDefender(BaseDefender):
    """
    Purifier defense from "Purifier: Defending Data Inference Attacks via Transforming Confidence Scores"
    (Yang et al., AAAI 2023, https://ojs.aaai.org/index.php/AAAI/article/view/26289). Use the arXiv version
    (https://arxiv.org/abs/2212.00612): its appendix has the architectures and training settings.

    Where the paper is silent or ambiguous, this implementation chooses as follows. The numbers come from
    an overfit CIFAR-10 ResNet-18 (99.75% train / 45.5% test accuracy), checked in October 2026.

    - Output: log-probabilities, so the attacks' softmax gives back the purified vector p exactly.

    - Reformer G noise: fixed-scale Gaussian noise on a batch-normalised latent (no learnable scale).
      Eq. 1 has no KL term, and without it a learned variance shrinks the noise away (latent std
      1 -> 5e-6 in one epoch). A fixed scale on an unnormalised latent fails too: the latent grows until
      the noise is negligible. Adding a KL term makes the decoder ignore the latent instead (one output per
      predicted label). Batch-normalising the latent fixes its scale, so the noise keeps its effect.
      The appendix also uses batch normalization in G's hidden layers.

    - Reformer loss: per-sample squared L2 reconstruction ("L2 norm") + lambda * cross-entropy, with
      lambda = 0.1 by default. With lambda = 1 the cross-entropy dominates and G outputs one-hot vectors
      (every non-member top-1 = 1.000), which turns Purifier into one-hot encoding. The paper's Fig. 7 shows
      spread-out confidences after purification instead.

    - Repeated queries: the latent noise of each query is seeded by a hash of the quantized input, so the
      same input always gets the same output (like MemGuard's one-time randomness). With fresh noise per
      query, labels changed over 20 repeated queries for 89% of swapped members and 12% of test
      non-members, against 0.3% of the other members: a replay signal the paper says Purifier defends.

    - Label swapper H: a random p_swap = (acc_train - acc_test) / acc_train share of the members get the
      second largest label of p, as in the paper. acc_test is estimated on the val (reference) split,
      because the test split holds the auditing non-members.

    - Swap-set lookup: kNN on the classifier's confidences c, not on the purified p, as in the paper ("H checks
      if c has a match in P_index ... and swaps the label of p"). This is deliberate: p carries the reformer's
      noise, so it could not match exactly. The match uses log-confidences rather than probabilities. Softmax
      saturates on confident predictions, so distinct inputs can share the same probability vector: with
      probabilities, 63.5% of non-members matched a member at the old threshold. Log-confidences of the same
      input differ by at most ~1e-5 across batch compositions and those of distinct inputs by at least ~2.4.

    - Reformer shape: the appendix's, see ConfidenceReformer. Defaults are its CIFAR-10 settings.
    """
    def __init__(self, defender_configs: DefenderConfigs):
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
        self.logger.print_it("PurifierDefender: training original model...")
        return super().train_model(train_configs=train_configs, return_stats=return_stats)

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.device = device
        self.logger.print_it("PurifierDefender: starting training of purifier module...")
        self.train_reformer()
        self.logger.print_it("PurifierDefender: finished training reformer, building Pindex for label swapping...")
        self.label_swapper.set_Pindex(self.build_pindex())
        self.label_swapper.set_device(self.device)
        self.logger.print_it('PurifierDefender: building defended model to be used for future prediction...')
        self.defended_model = PurifiedModelWrapper(original_model=self.trained_model,
                                                reformer=self.reformer,
                                                label_swapper=self.label_swapper,
                                                threshold=self.purifier_configs.swap_threshold,
                                                noise_quantization=self.purifier_configs.noise_quantization).to(self.device)
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
                if inputs.size(0) < 2:
                    continue  # the reformer's batch-normalised latent cannot train on a single sample
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    c = torch.nn.functional.softmax(self.original_model(inputs), dim=-1)
                    l = torch.nn.functional.one_hot(c.argmax(dim=-1), num_classes=c.size(1)).float()
                c_hat = self.reformer(c, l)  # fresh noise in training; inference seeds it per query
                # Eq. 1 with per-sample squared L2; see the class docstring for why lambda defaults to 0.1
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
    """
    PURIFIER confidence reformer G_theta.
    Gaussian noise of fixed scale is added to a batch-normalised latent, in training and at
    inference. A learned or unnormalised latent scale lets training shrink the noise away
    (see PurifierDefender's docstring).

    Shape from the paper's appendix: layers [2K, h, 2h, 4h, latent, 4h, 2h, h, K], with batch
    normalization and ReLU in the hidden layers. The appendix's settings per dataset:
        dataset                         h    latent  epochs        lr
        CIFAR10 (the defaults here)     32   2       100           0.01
        UTKFace                         32   2       300           1e-4
        Purchase100, Texas, Location    128  20      150/100/100   1e-4
        FaceScrub530, CIFAR100          512  100     300           5e-4
    The appendix's output sizes for CIFAR10 (20) and FaceScrub530 (1060) do not match K;
    UTKFace (2) and Purchase100 (100) do, so the output here is K.
    """
    def __init__(self, num_classes, latent_dim=2, hidden=32, noise_sigma=1.0):
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
            torch.nn.BatchNorm1d(latent_dim, affine=False),
        )

        # Decoder: latent || onehot(label) -> 4h -> 2h -> h -> K
        self.dec = torch.nn.Sequential(
            *hidden_layer(latent_dim + num_classes, 4 * hidden),
            *hidden_layer(4 * hidden, 2 * hidden),
            *hidden_layer(2 * hidden, hidden),
            torch.nn.Linear(hidden, num_classes),
        )
        self.sm = torch.nn.Softmax(dim=-1)
    
    def forward(self, c, l_onehot, noise=None):
        """
        noise: optional (B, latent_dim) standard Gaussian draws; fresh draws are used when None
        """
        r = self.enc(torch.cat([c, l_onehot], dim=-1))
        if noise is None:
            noise = torch.randn_like(r)
        z = r + self.noise_sigma * noise

        dec_in = torch.cat([z, l_onehot], dim=-1)
        return self.sm(self.dec(dec_in))



class TorchLabelSwapper(torch.nn.Module):
    """
    Pure PyTorch kNN lookup of the members whose label is swapped.
    Computes nearest neighbors fully on GPU.
    """
    def __init__(self, Pindex: Union[torch.Tensor, list] = None):
        """
        Pindex: list of (K,) tensors OR a (T,K) tensor of swap-set log-confidences
        """
        super().__init__()
        self.P = None
        if Pindex is not None:
            self.set_Pindex(Pindex)

    def set_Pindex(self, Pindex: Union[torch.Tensor, list]):
        if isinstance(Pindex, list):
            Pindex = torch.stack(Pindex)            # (T, K)
        self.P = Pindex.float()

    def set_device(self, device: torch.device):
        self.device = device
        if self.P is not None:
            self.P = self.P.to(device)

    @torch.no_grad()
    def query(self, log_C_batch, threshold):
        """
        log_C_batch: (B, K) log-confidence vectors
        Returns: BoolTensor(B,), True where the query matches a swap-set member
        """
        assert self.P is not None, "TorchLabelSwapper: Pindex must be provided to use the query function!"

        log_C_batch = log_C_batch.float().to(self.device)  # (B, K)
        if self.P.size(0) == 0:
            return torch.zeros(log_C_batch.size(0), dtype=torch.bool, device=self.device)

        # 1-NN on exact L2 distances: the ||a||^2 + ||b||^2 - 2 a·b expansion loses far more
        # precision than the threshold on log-confidences of large magnitude
        dist = torch.cdist(log_C_batch, self.P, compute_mode='donot_use_mm_for_euclid_dist')  # (B,T)
        return dist.min(dim=1).values <= threshold

    def forward(self, inputs, threshold):
        if self.training:
            raise RuntimeError("TorchLabelSwapper: forward should not be used in training mode!")
        return self.query(inputs, threshold=threshold)


class PurifiedModelWrapper(torch.nn.Module):
    """
    Drop-in model wrapper for MI benchmarking.
    """
    def __init__(self, original_model: torch.nn.Module, reformer: ConfidenceReformer, label_swapper: TorchLabelSwapper, threshold: float, noise_quantization: float):
        super().__init__()
        self.original_model = original_model
        self.reformer = reformer
        self.label_swapper = label_swapper
        self.threshold = threshold
        self.noise_quantization = noise_quantization

    def query_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard Gaussian latent noise seeded by a hash of each quantized query, so a repeated query
        gets the same purified output (see PurifierDefender's docstring). Same scheme as MemGuard's
        one-time randomness: independent of batch order and of the global RNG, and queries within
        the quantization width of each other share their noise.
        x: (B, ...) model inputs
        returns (B, latent_dim) noise on x's device
        """
        # Transfer first, then cast: MPS can attempt the dtype conversion on
        # the source device when .to(device='cpu', dtype=float64) is combined.
        quantized = (x.detach().cpu().to(dtype=torch.float64) / self.noise_quantization).round().to(torch.int64)
        noise = []
        for row in quantized:
            digest = hashlib.sha256()
            digest.update(str(tuple(row.shape)).encode('ascii'))
            digest.update(row.numpy().astype('<i8', copy=False).tobytes())
            generator = torch.Generator().manual_seed(int.from_bytes(digest.digest()[:8], 'big'))
            noise.append(torch.randn(self.reformer.latent_dim, generator=generator))
        return torch.stack(noise).to(x.device)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("PurifierDefender: PurifiedModelWrapper should not be used in training mode!")
        # Log probabilities satisfy the repository logits contract: softmax
        # reconstructs the purified confidence vector without a second softmax.
        probabilities = self.defend(x)
        return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
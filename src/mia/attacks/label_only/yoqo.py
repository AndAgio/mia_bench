from typing import List, Tuple, Union
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms


# Samples whose queries are searched for together. Each sample keeps its own loss, stop and models, so the queries are
# those of a search one sample at a time.
SEARCH_BATCH_SIZE = 128
# The authors' code scales the loss of each in/out pair by CODE_ONLINE_SCALE / (1 + alpha) online, and the loss of each
# out-model, distance included, by CODE_OFFLINE_SCALE / (1 + gamma) offline (shadow_training/yoqo_atk.py). The scales
# are 1 at the paper's alpha and gamma, and the paper's experiments on alpha and gamma (Fig. 6) used them.
CODE_ONLINE_SCALE = 3.0
CODE_OFFLINE_SCALE = 6.0


class Yoqo(BaseMIA):
    """YOQO: a label-only attack that queries the target once per sample, with a query crafted on shadow models.

    Paper: "You Only Query Once: An Efficient Label-Only Membership Inference Attack" (ICLR 2024),
    https://tianweiz07.github.io/Papers/24-iclr-1.pdf.
    Documentation: docs/attacks/yoqo.md.
    For each sample, gradient descent from the sample finds a query that the shadow models without the sample label
    otherwise (online, the ones with it keep its label; offline, the query stays close to it). A sample is a member when
    the target keeps its label on the query: the score is the decision, 0 or 1.
    """
    binary_scores = True

    def __init__(self,
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        """Sample the shadow datasets and build one shadow model per dataset."""
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        assert self.shadow_configs.mode == self.attack_configs.mode, \
            "YOQO's shadow datasets and attack must have the same mode, online or offline."
        self.mode = self.attack_configs.mode
        self.logger.print_it(f"Working with YOQO in {self.mode.upper()} mode!")
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('YOQO attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    calibration_dataset=self.calibration_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('YOQO attacker: definition of shadow models...')
        self.shadow_manager.build_shadow_models(n_models=self.shadow_configs.n_shadow_datasets,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        """Train the shadow models, each on its own shadow dataset with the true labels."""
        self.logger.print_it('YOQO attacker: training all shadow models. This will take a while...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config, labels_mode='original')
        self.reset_logger()
        h, m, s = convert_to_hms(time.time() - start)
        self.logger.print_it(f'YOQO attacker: done training the shadow models. It took {h}:{m:02d}:{s:02d}...')

    def score(self, dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample 1 when the target keeps its label on the sample's query, 0 otherwise: one query each."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('YOQO attacker: searching for the queries...')
        start = time.time()
        queries, labels = self.search_queries(dataset, self.in_models_mask(dataset.get_indices(mode='original')),
                                              device)
        scores = self.query_target(queries, labels, device)
        h, m, s = convert_to_hms(time.time() - start)
        self.logger.print_it(f'YOQO attacker: done scoring the samples. It took {h}:{m:02d}:{s:02d}...')
        return [MembershipScores(scores, decisions=scores.copy())]

    def model_ids(self) -> List[int]:
        """Return the shadow models' ids, in the order of the masks' columns."""
        assert self.shadow_manager.get_n_models() == self.shadow_manager.get_n_datasets(), \
            'YOQO trains one shadow model per shadow dataset.'
        return list(self.shadow_manager.get_all_model_indeces())

    def in_models_mask(self, indices: List[int]) -> np.ndarray:
        """Return whether each shadow model trained on each sample: the in-models online, none offline."""
        ids = self.model_ids()
        mask = np.zeros((len(indices), len(ids)), dtype=bool)
        for row, index in enumerate(indices):
            inside = set(self.shadow_manager.find_all_in_dataset_indices_for_sample_id(id=index))
            mask[row] = [model_id in inside for model_id in ids]
        return mask

    def search_queries(self, dataset, in_mask: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return each sample's query, searched for in batches of samples, and the sample's label."""
        models = [self.shadow_manager.get_model(model_id).to(device).eval() for model_id in self.model_ids()]
        loader = DataLoader(dataset, batch_size=SEARCH_BATCH_SIZE, shuffle=False)
        queries, labels, done, start = [], [], 0, time.time()
        for x, y, _, _ in loader:
            h, m, s = convert_to_hms(time.time() - start)
            self.logger.print_it_same_line(f'YOQO attacker: searching for the queries of samples {done + 1}-'
                                           f'{done + len(x)} of {len(dataset)} [{h:02d}:{m:02d}:{s:02d}]...',
                                           console_only=True)
            inside = torch.as_tensor(in_mask[done:done + len(x)], device=device)
            queries.append(self.search_batch(models, x.to(device), y.to(device), inside).cpu())
            labels.append(y)
            done += len(x)
        self.logger.set_logger_newline(console_only=True)
        return torch.cat(queries), torch.cat(labels)

    @staticmethod
    @torch.no_grad()
    def target_labels(models: List[torch.nn.Module], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return, for each sample and model, the model's top label other than the sample's (l'_i, Sec. 3.2)."""
        top2 = torch.stack([model(x) for model in models], dim=1).topk(2, dim=2).indices
        return torch.where(top2[..., 0] == y[:, None], top2[..., 1], top2[..., 0])

    def search_batch(self, models: List[torch.nn.Module], x0: torch.Tensor, y: torch.Tensor,
                     inside: torch.Tensor) -> torch.Tensor:
        """Return each sample's query: gradient descent from the sample on the paper's Eq. 4 (online) or Eq. 5.

        Online, the in-models should keep the sample's label and the out-models predict their l'_i; offline, the
        out-models predict l'_i while the query stays close to the sample. The losses are scaled as in the authors'
        code, the distance counted with each out-model. Each step follows the gradient at the current query. A sample's
        search stops once its loss, divided by its number of in/out pairs or out-models, is at most the threshold, or
        after the most iterations.
        """
        configs = self.attack_configs
        targets = self.target_labels(models, x0, y)
        outside = ~inside
        n_pairs = outside.sum(dim=1).clamp(min=1)
        x = x0.clone()
        active = torch.ones(len(x), dtype=torch.bool, device=x.device)
        for _ in range(configs.adv_opt_max_iter):
            rows = active.nonzero().squeeze(1)
            query = x[rows].requires_grad_(True)
            losses = torch.zeros(len(rows), dtype=query.dtype, device=x.device)
            gradient = torch.zeros_like(query)
            # One backward pass per model, so that only one model's graph is held at a time.
            for column, model in enumerate(models):
                logits = model(query)
                out_loss = F.cross_entropy(logits, targets[rows, column], reduction='none')
                if self.mode == 'online':
                    in_loss = F.cross_entropy(logits, y[rows], reduction='none')
                    term = (inside[rows, column] * in_loss + configs.alpha * outside[rows, column] * out_loss) \
                        * CODE_ONLINE_SCALE / (1 + configs.alpha)
                else:
                    distance = ((query - x0[rows]) ** 2).flatten(1).mean(dim=1)
                    term = outside[rows, column] * (configs.gamma * distance + out_loss) \
                        * CODE_OFFLINE_SCALE / (1 + configs.gamma)
                gradient += torch.autograd.grad(term.sum(), query)[0]
                losses += term.detach()
            x[rows] = query.detach() - configs.adv_opt_lr * gradient
            # As in the authors' code, the step is taken before the stop: the query is one step past the threshold.
            active[rows] = losses / n_pairs[rows] > configs.adv_opt_loss_threshold
            if not active.any():
                break
        return x

    @torch.no_grad()
    def query_target(self, queries: torch.Tensor, labels: torch.Tensor, device: torch.device) -> np.ndarray:
        """Send each query to the target once, and return 1 where it predicts the sample's label, 0 otherwise."""
        target = self.defender_model.to(device).eval()
        kept = []
        for query, label in DataLoader(TensorDataset(queries, labels), batch_size=SEARCH_BATCH_SIZE, shuffle=False):
            kept.append((target(query.to(device)).argmax(dim=1).cpu() == label).numpy())
        return np.concatenate(kept).astype(np.int64)

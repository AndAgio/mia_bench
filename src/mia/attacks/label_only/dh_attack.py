"""DHAttack: a label-only attack comparing a sample's distance to a fixed input on the target and on shadow models."""
from typing import Iterable, Optional, Tuple, Union
import dataclasses
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from scipy.stats import norm
from src.data.helpers import MultiDatasets, TargetOverrideDataset, get_dataset_transform
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms
from src.utils.variables import BINARY_FEATURE_DATASETS


# Samples scored together, fewer for large inputs so that a batch holds at most QUERY_BATCH_VALUES input values. Each
# sample's queries depend on that sample only, so this sets the speed only.
QUERY_BATCH_SIZE = 256
QUERY_BATCH_VALUES = 2 ** 24
# Random images the threshold is set on, and the percentile of their relScores it is set at (paper Sec. 5.3, Fig. 13).
N_SYNTHETIC_SAMPLES = 200
SYNTHETIC_PERCENTILE = 98


class DHAttack(BaseMIA):
    """Label-only attack: a member's distance to a fixed input is larger on the target than on shadow models.

    Paper: "Enhanced Label-Only Membership Inference Attacks with Fewer Queries",
    https://www.usenix.org/system/files/usenixsecurity25-li-hao.pdf.
    Documentation: docs/attacks/dh.md.
    The --n_shadows shadow models all train on one shadow dataset, relabelled by the target, each with its own seed.
    A sample's fixedBD is the number of steps toward a fixed input after which a model's label changes; its relScore
    is the CDF of its fixedBD on the target under a Gaussian fitted to its fixedBDs on the shadow models. The score
    is -log(1 - relScore): the same order, without rounding to 1 near the top.
    """
    def __init__(self,
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        """Sample the shadow dataset and build the shadow models, which all train on it."""
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with DHAttack!")
        assert self.shadow_configs.mode == 'offline', \
            f"DHAttack uses offline shadow models, got mode={self.shadow_configs.mode}."
        assert self.shadow_configs.n_shadow_datasets == 1, \
            f"DHAttack trains its shadow models on one shadow dataset, got {self.shadow_configs.n_shadow_datasets}."
        self.attack_threshold = None
        self.fixed_point = None
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('DHAttack: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    calibration_dataset=self.calibration_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('DHAttack: definition of shadow models...')
        self.shadow_manager.add_dataset_replicas(index=0,
                                                n_replicas=self.attack_configs.n_models)
        self.shadow_manager.build_shadow_models(n_models=self.attack_configs.n_models,
                                                model_configs=self.model_configs)

    @property
    def is_image(self) -> bool:
        """Return whether the samples are images."""
        return self.base_dataset_configs.im_size is not None

    @property
    def has_own_decisions(self) -> bool:
        """Return whether the attack decides by itself: on images only, whose threshold is set on random images.

        On tabular data the attack rule falls back to the calibrated one, whose calibration samples the base class
        then draws.
        """
        return self.is_image

    @staticmethod
    def shadow_train_configs(train_config: TrainConfigs, index: int) -> TrainConfigs:
        """Return the training configs of shadow model index: the attacker's, with the seed moved by the index.

        The shadow models share their dataset, so their seeds are what makes their batches, augmentation and other
        training randomness differ, as between the paper's independently trained shadow models.
        """
        if train_config.seed is None:
            return train_config
        return dataclasses.replace(train_config, seed=train_config.seed + index)

    def optimize(self, train_config: TrainConfigs):
        """Relabel the shadow dataset with the target, train the shadow models on it, and set the threshold."""
        self.logger.print_it('DHAttack: relabelling shadow dataset based on target model...')
        start = time.time()
        distilled_dataset = self.relabel_shadow_dataset(train_config=train_config)
        h, m, s = convert_to_hms(time.time() - start)
        self.logger.print_it('DHAttack: done relabelling shadow dataset. It took {}:{:02d}:{:02d}...'.format(h, m, s))

        self.logger.print_it('DHAttack: training shadow models with distilled dataset. This will take a while...')
        start = time.time()
        wrapped_dataset = MultiDatasets([distilled_dataset], ids=['train'])
        for i in range(self.attack_configs.n_models):
            self.shadow_manager.train_single_model_on_given_dataset(id=i,
                                                                    train_configs=self.shadow_train_configs(
                                                                        train_config, i),
                                                                    dataset=wrapped_dataset)
        self.reset_logger()
        h, m, s = convert_to_hms(time.time() - start)
        self.logger.print_it('DHAttack: done training the shadow models on the distilled dataset. '
                             'It took {}:{:02d}:{:02d}...'.format(h, m, s))
        if not self.is_image:
            self.logger.print_it('DHAttack: no threshold of its own on tabular data; its decisions come from the '
                                 'calibrated rule.')
            return
        self.logger.print_it('DHAttack: finding the threshold on synthetic samples...')
        start = time.time()
        self.find_optimal_threshold(device=train_config.device)
        h, m, s = convert_to_hms(time.time() - start)
        self.logger.print_it('DHAttack: done finding the threshold on synthetic samples. '
                             'It took {}:{:02d}:{:02d}...'.format(h, m, s))

    def relabel_shadow_dataset(self, train_config: TrainConfigs):
        """Return the shadow dataset with the labels the target predicts for it (the paper's Sec. 3.2)."""
        shadow_dataset = self.shadow_manager.get_dataset(index=0, labels='original')
        shadow_loader = torch.utils.data.DataLoader(shadow_dataset, batch_size=train_config.batch_size, shuffle=False)
        all_relabels = []
        device = self.get_device(train_config.device)
        self.defender_model.to(device)
        with torch.no_grad():
            for batch_index, (data, original_labels, _, _) in enumerate(shadow_loader):
                self.logger.print_it_same_line(f"DHAttack: relabelling batch {batch_index+1}/{len(shadow_loader)}...",
                                               console_only=True)
                data = data.to(device)
                preds = torch.argmax(self.defender_model(data), dim=1).cpu()
                all_relabels.append(preds)
        self.logger.set_logger_newline(console_only=True)
        all_relabels = torch.cat(all_relabels, dim=0)
        return TargetOverrideDataset(shadow_dataset, all_relabels)

    def score(self, dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample with -log(1 - relScore), and threshold it on images."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('DHAttack attacker: scoring the samples...')
        start = time.time()
        scores, decisions = self.infer_dataset(dataset=dataset, device=device)
        h, m, s = convert_to_hms(time.time() - start)
        self.logger.print_it(f'DHAttack attacker: done scoring the samples. It took {h}:{m:02d}:{s:02d}...')
        assert scores.shape == (len(dataset),), f"Unexpected shape for scores: {scores.shape}"
        assert decisions is None or decisions.shape == scores.shape, \
            f"Unexpected shape for decisions: {decisions.shape}"
        return [MembershipScores(scores, decisions=decisions)]

    @torch.no_grad()
    def fixedbd_over_model(self, model: torch.nn.Module, inputs: torch.Tensor,
                           true_labels: torch.Tensor) -> np.ndarray:
        """Return each sample's fixedBD on the model: the first of K steps toward the fixed input at which the model's
        label is not the sample's, K if there is none (the paper's Alg. 2).

        Each step queries only the samples whose label has not changed yet, so a sample costs the same queries as
        when it is scored alone. The paper's last step, the fixed input itself, is not queried: d is K either way.
        """
        n_queries = self.attack_configs.n_queries
        true_labels = true_labels.to(inputs.device)
        x_diff = self.construct_fixed_input(inputs) - inputs
        distances = torch.full((inputs.shape[0],), float(n_queries), device=inputs.device)
        active = torch.arange(inputs.shape[0], device=inputs.device)
        for k in range(n_queries):
            if active.numel() == 0:
                break
            x_masked = k / n_queries * x_diff[active] + inputs[active]
            changed = torch.argmax(model(x_masked), dim=1) != true_labels[active]
            distances[active[changed]] = k
            active = active[~changed]
        return distances.cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def rel_score(self, target_model: torch.nn.Module, inputs: torch.Tensor,
                  true_labels: torch.Tensor) -> np.ndarray:
        """Return each sample's -log(1 - relScore), relScore being the CDF of its fixedBD on the target under the
        Gaussian of its fixedBDs on the shadow models (the paper's Alg. 1, with their mean and population std).

        The CDF itself is exactly 1.0 for every sample more than about 8.3 standard deviations above the shadow
        models' mean, and for every sample above it when they all agree (their std is 0, plus 1e-30 as in the
        authors' code), so that no threshold separates those samples. -log(1 - CDF) orders the samples as the CDF does,
        and keeps them apart.
        """
        shadow_distances = np.stack([self.fixedbd_over_model(model=shadow_model, inputs=inputs,
                                                             true_labels=true_labels)
                                     for shadow_model in self.shadow_manager.get_all_models().values()], axis=1)
        target_distances = self.fixedbd_over_model(model=target_model, inputs=inputs, true_labels=true_labels)
        return -norm.logsf(target_distances, loc=shadow_distances.mean(axis=1),
                           scale=shadow_distances.std(axis=1) + 1e-30)

    def construct_fixed_input(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the fixed input each sample moves toward.

        Images move toward one image (white: every pixel 255, the paper's RGB-255; black: 0; random: one image of
        uniform random pixels), through the dataset's transform. Tabular samples move toward a point with every
        feature at the sample's largest value, as the paper does for non-image data (Sec. 6.1), or, for black, at
        its smallest; random is one random record, as for images.
        """
        mode = self.attack_configs.fixed_input_mode
        if not self.is_image and mode in ('white', 'black'):
            dims = tuple(range(1, inputs.dim()))
            extreme = inputs.amax(dim=dims, keepdim=True) if mode == 'white' else inputs.amin(dim=dims, keepdim=True)
            return extreme.expand_as(inputs)
        if self.fixed_point is None:
            shape = tuple(inputs.shape[1:])
            self.fixed_point = self.build_fixed_image(shape) if self.is_image else self.build_random_record(shape)
        return self.fixed_point.to(inputs.device).unsqueeze(0).expand_as(inputs)

    def build_random_record(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Return the random record of the random mode: each feature drawn uniformly from its values, like the
        pixels of the random image: 0 or 1 for binary datasets, [0, 1] otherwise (the range of the tabular datasets).
        """
        mode = self.attack_configs.fixed_input_mode
        if mode != 'random':
            raise ValueError(f"Unsupported fixed input mode {mode!r} for DHAttack: use 'white', 'black' or 'random'.")
        rng = np.random.default_rng(seed=12345)
        if self.base_dataset_configs.name in BINARY_FEATURE_DATASETS:
            values = rng.integers(0, 2, size=shape)
        else:
            values = rng.random(size=shape)
        return torch.tensor(values, dtype=torch.float32)

    def build_fixed_image(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Return the fixed image of the mode, of the given (C, H, W) shape, as the dataset transforms it."""
        C, H, W = shape
        mode = self.attack_configs.fixed_input_mode
        if mode == 'white':
            fixed_x_np = np.full((H, W, C), 255, dtype=np.uint8)
        elif mode == 'black':
            fixed_x_np = np.full((H, W, C), 0, dtype=np.uint8)
        elif mode == 'random':
            rng = np.random.default_rng(seed=12345)
            fixed_x_np = rng.integers(0, 256, size=(H, W, C), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported fixed input mode {mode!r} for DHAttack: use 'white', 'black' or 'random'.")
        return self.dataset_transform()(Image.fromarray(fixed_x_np.squeeze(axis=2) if C == 1 else fixed_x_np))

    def dataset_transform(self):
        """Return the transform of the shadow dataset, which turns PIL images into model inputs."""
        transform = get_dataset_transform(self.shadow_manager.get_dataset(index=0, labels='original'))
        if transform is None:
            raise RuntimeError("Could not find the input transform for the DHAttack shadow dataset.")
        return transform

    def construct_random_samples(self, shape: Tuple[int, ...], n_samples: int) -> torch.Tensor:
        """Return n_samples images of uniform random pixels, drawn from the attacker's seed, as the dataset
        transforms them."""
        C, H, W = shape
        transform = self.dataset_transform()
        rng = np.random.default_rng(seed=self.seed)
        samples = []
        for _ in range(n_samples):
            random_x_np = rng.integers(0, 256, size=(H, W, C), dtype=np.uint8)
            samples.append(transform(Image.fromarray(random_x_np.squeeze(axis=2) if C == 1 else random_x_np)))
        return torch.stack(samples, dim=0)

    def find_optimal_threshold(self, device: Union[torch.device, str] = 'cpu'):
        """Set the threshold at a high percentile of the scores of random images (the paper's Sec. 5.3)."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        dummy_sample = self.shadow_manager.get_dataset(index=0, labels='original')[0][0]
        synthetic_samples = self.construct_random_samples(shape=tuple(dummy_sample.shape),
                                                          n_samples=N_SYNTHETIC_SAMPLES)
        batch_size = self.batch_size(synthetic_samples)
        # A random image belongs to no class: its fixedBD is measured from the label the target predicts for it, as
        # UBA measures its random images' distances. The paper gives no label (Sec. 5.3).
        self.defender_model.to(device)
        self.defender_model.eval()
        with torch.no_grad():
            synthetic_labels = torch.cat([torch.argmax(self.defender_model(batch.to(device)), dim=1).cpu()
                                          for batch in synthetic_samples.split(batch_size)])
        rel_scores_synthetic = self.score_batches(self.defender_model,
                                                  zip(synthetic_samples.split(batch_size),
                                                      synthetic_labels.split(batch_size)),
                                                  device)
        self.attack_threshold = np.percentile(rel_scores_synthetic, q=SYNTHETIC_PERCENTILE)
        self.logger.print_it(f"DHAttack attacker: threshold set at {self.attack_threshold:.4f}.")

    @staticmethod
    def batch_size(inputs: torch.Tensor) -> int:
        """Return how many samples like those of inputs to score together."""
        values = int(np.prod(inputs.shape[1:]))
        return max(1, min(QUERY_BATCH_SIZE, QUERY_BATCH_VALUES // max(1, values)))

    @torch.no_grad()
    def score_batches(self, target_model: torch.nn.Module, batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
                      device: torch.device) -> np.ndarray:
        """Return the scores of the samples of the batches, in their order."""
        for model in [target_model, *self.shadow_manager.get_all_models().values()]:
            model.to(device)
            model.eval()
        scores = []
        start = time.time()
        for batch_id, (batch_x, batch_y) in enumerate(batches):
            h, m, s = convert_to_hms(time.time() - start)
            self.logger.print_it_same_line(f"DHAttack attacker: scoring batch {batch_id+1} "
                                           f"[{h:02d}:{m:02d}:{s:02d}]...", console_only=True)
            scores.append(self.rel_score(target_model, batch_x.to(device), batch_y.to(device)))
        self.logger.set_logger_newline(console_only=True)
        return np.concatenate(scores)

    def decide(self, scores: np.ndarray) -> Optional[np.ndarray]:
        """Return whether each score is above the threshold, None without one (tabular data)."""
        if self.attack_threshold is None:
            assert not self.is_image, "Call find_optimal_threshold(...) before scoring images with DHAttack."
            return None
        return (scores > self.attack_threshold).astype(np.int64)

    @torch.no_grad()
    def infer_dataset(self, dataset: torch.utils.data.Dataset, device: Union[torch.device, str] = 'cpu'):
        """Return the scores of the dataset's samples on the target, and their decisions (None without a
        threshold)."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        batch_size = self.batch_size(dataset[0][0].unsqueeze(0))
        batches = ((batch_x, batch_y) for batch_x, batch_y, _, _ in
                   DataLoader(dataset, batch_size=batch_size, shuffle=False))
        scores = self.score_batches(self.defender_model, batches, device)
        return scores, self.decide(scores)

    @torch.no_grad()
    def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor,
                    device: Union[torch.device, str] = 'cpu'):
        """Return the scores of a batch on the given model as the target, and their decisions."""
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        scores = self.score_batches(model, [(batch_x, batch_y)], device)
        return scores, self.decide(scores)

    @torch.no_grad()
    def infer_single(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                     device: Union[torch.device, str] = 'cpu'):
        """Return one sample's score on the given model as the target, and its decision."""
        scores, decisions = self.infer_batch(model, x.unsqueeze(0), y.unsqueeze(0), device)
        return scores.item(), None if decisions is None else decisions.item()

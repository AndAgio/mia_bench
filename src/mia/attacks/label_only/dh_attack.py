
from tkinter import Image
from typing import Union
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from scipy.stats import norm
from sklearn.metrics import accuracy_score, roc_curve
from src.data.helpers import MultiDatasets, MyConcatDataset
from src.mia.attacks.base_mia import BaseMIA
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms
import copy


class DHAttack(BaseMIA):
    # Implementation of DHAttack of "Enhanced Label-Only Membership Inference Attacks with Fewer Queries" (https://www.usenix.org/system/files/usenixsecurity25-li-hao.pdf).
    def __init__(self, 
                defender_model: torch.nn.Module,
                defender_dataset: MultiDatasets,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, defender_dataset=defender_dataset, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with DHAttack!")
        assert self.shadow_configs.mode == 'offline', f"DHAttack should be used with offline shadow models, but found mode={self.shadow_configs.mode} instead!"
        if self.shadow_configs.n_shadow_datasets != 1:
            self.logger.print_it(f"DHAttack [WARNING]: when using DHAttack, only 1 shadow dataset must be used! Modifying shadow_configs on the fly to set n_shadow_datasets to 1.")
            self.shadow_configs.n_shadow_datasets = 1
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('DHAttack: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(original_datasets=self.defender_dataset,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('DHAttack: definition of shadow models...')
        self.shadow_manager.add_dataset_replicas(index=0, 
                                                n_replicas=self.attack_configs.n_models)
        self.shadow_manager.build_shadow_models(n_models=self.attack_configs.n_models,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('DHAttack: relabelling shadow dataset based on target model...')
        start = time.time()
        distilled_dataset = self.relabel_shadow_dataset(train_config=train_config)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('DHAttack: done relabelling shadow dataset. It took {}:{:02d}:{:02d}...'.format(h, m, s))

        self.logger.print_it('DHAttack: training shadow models with distilled dataset. This will take a while. Sit back and chill...')
        start = time.time()
        wrapped_dataset = MultiDatasets([distilled_dataset], ids=['train'])
        for i in range(self.attack_configs.n_models):
            self.shadow_manager.train_single_model_on_given_dataset(id=i,
                                                                    train_configs=train_config,
                                                                    dataset=wrapped_dataset)
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('DHAttack: done optimizing shadow models over the distilled dataset. It took {}:{:02d}:{:02d}...'.format(h, m, s))

        self.logger.print_it('DHAttack: finding optimal threshold on synthetic samples...')
        start = time.time()
        self.find_optimal_threshold(device=train_config.device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('DHAttack: done finding optimal threshold on synthetic samples. It took {}:{:02d}:{:02d}...'.format(h, m, s))

    def relabel_shadow_dataset(self, train_config: TrainConfigs):
        # Relabel the shadow dataset according to the predictions of the target model, as done in the transfer attack of "Label-Only Membership Inference Attacks" (https://proceedings.mlr.press/v139/choquette-choo21a/choquette-choo21a.pdf).
        shadow_dataset = self.shadow_manager.get_dataset(index=0, labels='original')
        shadow_loader = torch.utils.data.DataLoader(shadow_dataset, batch_size=train_config.batch_size, shuffle=False)
        all_relabels = []
        device = self.get_device(train_config.device)
        self.defender_model.to(device)
        with torch.no_grad():
            for batch_index, (data, original_labels, _, _) in enumerate(shadow_loader):
                self.logger.print_it_same_line(f"DHAttack: relabelling batch {batch_index+1}/{len(shadow_loader)}...", console_only=True)
                data = data.to(device)
                preds = torch.argmax(self.defender_model(data), dim=1).cpu()
                all_relabels.append(preds)
        self.logger.set_logger_newline(console_only=True)
        all_relabels = torch.cat(all_relabels, dim=0)
        # Relabeling the shadow dataset with the obtained relabels
        distilled_dataset = copy.deepcopy(shadow_dataset)
        try:
            distilled_dataset.set_targets(all_relabels)
        except AttributeError:
            distilled_dataset.targets = all_relabels
        return distilled_dataset

    def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('DHAttack attacker: measuring attack effectiveness...')
        start = time.time()
        audit_dataset = self.audit_manager.get(labels='original')
        scores, decisions = self.infer_dataset(dataset=audit_dataset,
                                                device=device)
        print(f"scores: {scores}")
        print(f"decisions: {decisions}")
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('DHAttack attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        metrics = self.compute_stats(scores)
        mia_audit_dataset = self.audit_manager.get(labels='mia')
        mia_labels = torch.asarray([label for _, (_, label, _, _) in enumerate(mia_audit_dataset)])
        assert scores.shape == mia_labels.shape, f"Unexpected shape for scores: {scores.shape}, expected {mia_labels.shape}"
        assert decisions.shape == mia_labels.shape, f"Unexpected shape for decisions: {decisions.shape}, expected {mia_labels.shape}"
        correct_decisions = (decisions == mia_labels)
        print(correct_decisions)
        attack_accuracy = np.mean(correct_decisions.numpy())
        self.logger.print_it(f"DHAttack attacker: attack accuracy at threshold {self.attack_threshold:.4f} is {attack_accuracy:.4f}.")
        return metrics

    @torch.no_grad()
    def fixedbd_over_model(self, model: torch.nn.Module, inputs: torch.Tensor, true_labels: torch.Tensor) -> np.ndarray:
        device = inputs.device
        model.eval()
        x_fixed = self.construct_fixed_input(inputs).to(device)
        true_labels = true_labels.to(device)
        x_diff = x_fixed - inputs
        distances = torch.ones(inputs.shape[0], device=device) * self.attack_configs.n_queries
        mask = torch.ones(inputs.shape[0], device=device, dtype=torch.bool)
        for k in range(self.attack_configs.n_queries):
            x_masked = k/self.attack_configs.n_queries * x_diff + inputs
            # self.debug_image(x_masked[0])
            logits = model(x_masked)
            predictions = torch.argmax(logits, dim=1)
            predictions_to_check = torch.where(mask, predictions, -1)
            labels_to_check = torch.where(mask, true_labels, -1)
            # predictions_to_check = predictions[torch.where(mask == True)]
            # current_wrong_predictions = torch.where(predictions_to_check != true_labels[torch.where(mask == True)])
            current_wrong_predictions = torch.where(predictions_to_check != labels_to_check)
            distances[current_wrong_predictions] = k
            mask[current_wrong_predictions] = False
            if all(mask == False):
                break
        # self.logger.print_it(f"DEBUG: Computed FixedDB distances for {inputs.shape[0]} samples in {k+1} queries per sample.")
        return distances.cpu().numpy()
    
    @torch.no_grad()
    def rel_score(self, inputs: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        device = inputs.device
        shadow_models = self.shadow_manager.get_all_models()
        # print(f"shadow_models: {shadow_models}")
        dist_matrix = np.zeros((inputs.shape[0], len(shadow_models)))
        for i, shadow_model in shadow_models.items():
            # self.logger.print_it_same_line(f"Computing FixedDB distance with shadow model {i+1}/{len(shadow_models)}...", console_only=True)
            shadow_model = shadow_model.to(device)
            shadow_model.eval()
            distances = self.fixedbd_over_model(model=shadow_model, inputs=inputs, true_labels=true_labels)
            dist_matrix[:, i] = distances
        # self.logger.set_logger_newline(console_only=True)
        # self.logger.print_it(f"Computing normal distribution parameters for REL scores based on shadow models...")
        shadow_means = np.mean(dist_matrix, axis=1)
        assert shadow_means.shape == (inputs.shape[0],), f"Unexpected shape for shadow_means: {shadow_means.shape}, expected ({inputs.shape[0]},)"
        shadow_stds = np.std(dist_matrix, axis=1)
        assert shadow_stds.shape == (inputs.shape[0],), f"Unexpected shape for shadow_stds: {shadow_stds.shape}, expected ({inputs.shape[0]},)"
        # self.logger.print_it(f"Computing FixedDB distance with defender model...")
        target_model = self.defender_model.to(device)
        target_model.eval()
        target_distances = self.fixedbd_over_model(model=target_model, inputs=inputs, true_labels=true_labels)
        # self.logger.print_it(f"Computing REL scores...")
        rel_scores = norm.cdf(target_distances, loc=shadow_means, scale=shadow_stds+1e-30)
        return torch.asarray(rel_scores)

    def construct_fixed_input(self, inputs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = inputs.shape
        dataset_transformer = self.shadow_manager.get_dataset(index=0, labels='original').transform
        if self.attack_configs.fixed_input_mode == 'white':
            fixed_x_np = np.full((H, W, C), 255, dtype=np.uint8)  # H, W, C
        elif self.attack_configs.fixed_input_mode == 'black':
            fixed_x_np = np.full((H, W, C), 0, dtype=np.uint8)
        elif self.attack_configs.fixed_input_mode == 'random':
            rng = np.random.default_rng(seed=12345)
            fixed_x_np = rng.integers(0, 256, size=(H, W, C), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported fixed_input_mode {self.attack_configs.fixed_input_mode} for DHAttack! Supported modes are: 'white' and 'black'.")
        fixed_x_pil = Image.fromarray(fixed_x_np)
        x_fixed = dataset_transformer(fixed_x_pil)
        x_fixed = x_fixed.unsqueeze(0)
        x_fixed_batch = x_fixed.repeat(B, 1, 1, 1)
        return x_fixed_batch
    
    def construct_random_samples(self, shape: tuple[int], n_samples: int) -> torch.Tensor:
        C, H, W = shape
        dataset_transformer = self.shadow_manager.get_dataset(index=0, labels='original').transform
        samples = []
        for _ in range(n_samples):
            random_x_np = np.random.randint(0, 256, (H, W, C), dtype=np.uint8)
            random_x_pil = Image.fromarray(random_x_np)
            random_x_torch = dataset_transformer(random_x_pil)
            samples.append(random_x_torch)
        return torch.stack(samples, dim=0)  # n_samples, C, H, W

    def find_optimal_threshold(self, device: Union[torch.device, str] = 'cpu'):
        self.logger.print_it('DHAttack attacker: finding optimal threshold for attack decisions based on synthetic samples...')
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        start = time.time()
        n_synthetic_samples = 200
        dummy_sample = self.shadow_manager.get_dataset(index=0, labels='original')[0][0]
        synthetic_samples = self.construct_random_samples(shape=dummy_sample.shape, n_samples=n_synthetic_samples)
        synthetic_labels = torch.zeros(n_synthetic_samples, dtype=torch.long)  # The true labels do not matter, since the samples are random noise and the attack is label-only
        synthetic_dataset = torch.utils.data.TensorDataset(synthetic_samples, synthetic_labels)
        dataloader = DataLoader(synthetic_dataset, batch_size=1, shuffle=False)
        rel_scores_synthetic = []
        for batch_id, (batch_x, batch_y) in enumerate(dataloader):
            h, m, s = convert_to_hms(time.time()-start)
            self.logger.print_it_same_line(f"DHAttack attacker: processing batch {batch_id+1}/{len(dataloader)} of synthetic samples for optimal threshold search [{h:02d}:{m:02d}:{s:02d}]. This may take a while...", console_only=True)
            score = self.rel_score(inputs=batch_x.to(device), true_labels=batch_y.to(device))
            rel_scores_synthetic.append(score.cpu())
        self.logger.set_logger_newline(console_only=True)
        rel_scores_synthetic = torch.cat(rel_scores_synthetic, dim=0).numpy()
        # rel_scores_synthetic = self.rel_score(inputs=synthetic_samples, true_labels=torch.zeros(n_synthetic_samples, dtype=torch.long, device=device))
        # print(f"rel_scores_synthetic: {rel_scores_synthetic}")
        optimal_threshold = np.percentile(rel_scores_synthetic, q=98)
        # print(f"optimal_threshold: {optimal_threshold}")
        self.attack_threshold = optimal_threshold
        h, m, s = convert_to_hms(time.time()-start)
        self.logger.print_it(f"DHAttack attacker: optimal threshold found at {optimal_threshold:.4f}. Time taken to find optimal threshold: {h:02d}:{m:02d}:{s:02d}.")
    
    # def find_optimal_threshold(self, device: Union[torch.device, str] = 'cpu'):
    #     self.attack_threshold = 0.01

    # TODO: Add method to find optimal threshold based on an extra shadow model and shadow dataset.
    # assignees: AndAgio.

    @torch.no_grad()
    def infer_dataset(self, dataset: torch.utils.data.Dataset, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_dataset(...) for Transfer MIA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        all_scores = []
        start = time.time()
        for batch_id, (batch_x, batch_y, _, _) in enumerate(dataloader):
            h, m, s = convert_to_hms(time.time()-start)
            self.logger.print_it_same_line(f"DHAttack attacker: processing batch {batch_id+1}/{len(dataloader)} for inference [{h:02d}:{m:02d}:{s:02d}]. This may take a while...", console_only=True)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            scores = self.rel_score(inputs=batch_x, true_labels=batch_y)
            all_scores.append(scores.cpu())
        self.logger.set_logger_newline(console_only=True)
        all_scores = torch.stack(all_scores).numpy()
        all_decisions = (all_scores > self.attack_threshold).astype(np.int64)
        h, m, s = convert_to_hms(time.time()-start)
        self.logger.print_it(f"DHAttack attacker: inference on auditing dataset completed. Time taken: {h:02d}:{m:02d}:{s:02d}.")
        return all_scores.squeeze(), all_decisions.squeeze()
    
    @torch.no_grad()
    def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_batch(...) for Transfer MIA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        scores = self.rel_score(inputs=batch_x, true_labels=batch_y).numpy()
        decisions = (scores > self.attack_threshold).astype(np.int64)
        return scores.cpu().numpy(), decisions

    @torch.no_grad()
    def infer_single(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_single(...) for Transfer MIA."
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        score = self.rel_score(inputs=x, true_labels=y).numpy()
        decision = (score > self.attack_threshold).astype(np.int64)
        return score.item(), decision.item()
    
    def debug_image(self, sample: torch.Tensor):
        import matplotlib.pyplot as plt
        sample = sample.cpu()
        mean = torch.tensor([x / 255.0 for x in [125.3, 123.0, 113.9]])
        std = torch.tensor([x / 255.0 for x in [63.0, 62.1, 66.7]])
        unnormalized_img = sample * std.view(3, 1, 1) + mean.view(3, 1, 1)
        # Now it is safe to clip values between 0 and 1, permute, and plot
        unnormalized_img = torch.clamp(unnormalized_img, 0, 1)
        plt.imshow(unnormalized_img.permute(1, 2, 0))
        plt.show()
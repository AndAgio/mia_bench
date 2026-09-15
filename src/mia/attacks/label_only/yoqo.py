from typing import Union
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.mia.attacks.base_mia import BaseMIA
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms


PROCESSING_BATCH_SIZE = 10


class Yoqo(BaseMIA):
    # Implementation of YOQO of "You only query once: An efficient label-only membership inference attack" (https://tianweiz07.github.io/Papers/24-iclr-1.pdf).
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        assert self.shadow_configs.mode == self.attack_configs.mode, f"Whenever working with YOQO the mode for shadow datasets and attack should be the same!"
        self.mode = self.attack_configs.mode
        self.logger.print_it(f"Working with YOQO in {self.mode.upper()} mode!")
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('YOQO attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('YOQO attacker: definition of shadow models...')
        self.shadow_manager.build_shadow_models(n_models=self.shadow_configs.n_shadow_datasets,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('YOQO attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('YOQO attacker: Done optimizing. It took {}:{:02d}:{:02d}...'.format(h, m, s))


    def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('Computing LiRA scores. This may take a while...')
        start = time.time()
        audit_dataset = self.audit_manager.get(labels='original')
        audit_dataset_indices = self.audit_manager.get_all_ids()
        trained_in_shadow_models_for_sample, trained_out_shadow_models_for_sample = self.get_in_out_model_indices_for_audit_samples(audit_dataset_indices=audit_dataset_indices)
        adversarial_examples = self.find_adversarial_examples(dataset=audit_dataset,
                                                                in_models_samples_map=trained_in_shadow_models_for_sample,
                                                                out_models_samples_map=trained_out_shadow_models_for_sample,
                                                                device=device)
        scores, decisions = self.compute_scores_from_adversaries(
            adversarial_examples=adversarial_examples,
            device=device,
        )
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('DHAttack attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
        metrics = self.compute_stats(scores, decisions=decisions)
        mia_audit_dataset = self.audit_manager.get(labels='mia')
        correct_decisions = (decisions == np.array([label for _, (_, label, _, _) in enumerate(mia_audit_dataset)]))
        attack_accuracy = np.mean(correct_decisions)
        self.logger.print_it(f"DHAttack attacker: attack accuracy at optimal threshold is {attack_accuracy:.4f}.")
        return metrics

    def compute_scores_from_adversaries(self, adversarial_examples: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        audit_dataset = self.audit_manager.get(labels='original')
        adversarial_examples_dataset = TensorDataset(adversarial_examples, torch.tensor([label for _, (_, label, _, _) in enumerate(audit_dataset)]))
        dataloader = DataLoader(adversarial_examples_dataset, batch_size=1, shuffle=False)
        scores = []
        decisions = []
        defender_model = self.defender_model.to(device)
        for batch_index, (adversarial_example, label) in enumerate(dataloader):
            defender_model.eval()
            with torch.no_grad():
                logits = defender_model(adversarial_example.to(device))
                pred_label = torch.argmax(logits, dim=1)
                print(f"DEBUG: Sample {batch_index+1}/{len(dataloader)}, original label: {label.item()}, adversarial example predicted label: {pred_label.item()}")    
            scores.append(logits[0, label.item()].item())
            if pred_label.item() != label.item():
                decisions.append(0)
            else:
                decisions.append(1)
        scores = np.array(scores)
        decisions = np.array(decisions)
        return scores, decisions

    def get_in_out_model_indices_for_audit_samples(self, audit_dataset_indices: list[int]):
        all_models_indices = self.shadow_manager.get_all_model_indeces()
        trained_in_shadow_models_for_sample = []
        trained_out_shadow_models_for_sample = []
        for index in audit_dataset_indices:
            in_models_indices = self.shadow_manager.find_all_in_dataset_indices_for_sample_id(id=index)
            out_models_indices = [index for index in all_models_indices if index not in in_models_indices]
            trained_in_shadow_models_for_sample.append(in_models_indices)
            trained_out_shadow_models_for_sample.append(out_models_indices)
        return trained_in_shadow_models_for_sample, trained_out_shadow_models_for_sample
    
    def find_adversarial_examples(self, dataset: torch.utils.data.Dataset, in_models_samples_map: list[list[int]], out_models_samples_map: list[list[int]], device: Union[torch.device, str] = 'cpu'):
        tot_samples = len(dataset)
        audit_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        assert self.shadow_manager.get_n_models() == self.shadow_manager.get_n_datasets(), "When working with YOQO the number of shadow models and shadow datasets should be the same!"
        start = time.time()
        self.logger.print_it(f"YOQO Attacker: Searching adversarial examples for all {tot_samples} samples. This may take a while...")
        adversarial_examples = []
        for sample_index, (sample, label, _, _) in enumerate(audit_loader):
            h, m, s = convert_to_hms(time.time()-start)
            self.logger.print_it_same_line(f"YOQO Attacker: Searching adversarial examples for sample {sample_index+1}/{tot_samples} [{h:02d}:{m:02d}:{s:02d}]. This may take a while...", console_only=True)
            out_models = out_models_samples_map[sample_index]
            in_models = in_models_samples_map[sample_index]
            adversarial_example = self.find_adversarial_example_for_sample(sample=sample,
                                                                            label=label,
                                                                            in_models=in_models,
                                                                            out_models=out_models,
                                                                            device=device)
            adversarial_examples.append(adversarial_example.squeeze(0).cpu())
        self.logger.set_logger_newline(console_only=True)
        adversarial_examples = torch.stack(adversarial_examples, dim=0)
        h, m, s = convert_to_hms(time.time()-start)
        self.logger.print_it(f"YOQO Attacker: Finished searching adversarial examples for all samples. It took {h:02d}:{m:02d}:{s:02d}.")
        return adversarial_examples

    def find_adversarial_example_for_sample(self, sample: torch.Tensor, label: torch.Tensor, in_models: list[Union[int, torch.nn.Module]], out_models: list[Union[int, torch.nn.Module]], device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        sample = sample.to(device)
        original_label = label.to(device)
        target_labels = self.find_target_labels_for_sample(sample=sample, original_label=original_label, out_models=out_models, device=device)

        sample.requires_grad = True
        original_sample = sample.clone().detach()
        print(f"sample.requires_grad: {sample.requires_grad}, original_sample.requires_grad: {original_sample.requires_grad}")

        adv_grad = None
        adv_loss = torch.tensor(float('inf'), device=device)
        iteration = 0
        ce_loss = torch.nn.CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss()
        while(adv_loss.item() > self.attack_configs.adv_opt_loss_threshold):
            
            ce_out_models_loss = 0
            for model_index, model in enumerate(out_models):
                if isinstance(model, torch.nn.Module):
                    pass
                elif isinstance(model, int):
                    model = self.shadow_manager.get_model(model)
                else:
                    raise ValueError('Model should be either a torch Module or an integer referring to the id of the shadow model!')
                model = model.to(device)
                model.eval()
                logits = model(sample)
                ce = ce_loss(logits, target_labels[model_index].unsqueeze(0))
                ce_out_models_loss += ce
            
            if self.attack_configs.mode == 'online':
                ce_in_models_loss = 0
                for model_index, model in enumerate(in_models):
                    if isinstance(model, torch.nn.Module):
                        pass
                    elif isinstance(model, int):
                        model = self.shadow_manager.get_model(model)
                    else:
                        raise ValueError('Model should be either a torch Module or an integer referring to the id of the shadow model!')
                    model = model.to(device)
                    model.eval()
                    logits = model(sample)
                    ce = ce_loss(logits, original_label)
                    ce_in_models_loss += ce
                total_loss = self.attack_configs.alpha * ce_out_models_loss + ce_in_models_loss
            elif self.attack_configs.mode == 'offline':
                mse_out_loss = 0
                mse_out_loss = mse_loss(sample, original_sample)
                total_loss = ce_out_models_loss + self.attack_configs.gamma * mse_out_loss
            else:
                raise ValueError(f"Unsupported mode '{self.attack_configs.mode}' for YOQO! Supported modes are 'online' and 'offline'.")

            total_loss.backward()
            grad = sample.grad.data

            adv_grad = grad.clone() if iteration == 0 else adv_grad + grad
            adv_loss = total_loss.clone()
            sample.grad.data.zero_()
            sample.data -= self.attack_configs.adv_opt_lr * adv_grad

            sample.grad.data.zero_()

            iteration += 1
            if iteration > self.attack_configs.adv_opt_max_iter:
                return sample.detach()
            
            print(f"DEBUG: Iteration {iteration}, total_loss: {total_loss.item():.4f}, ce_out_models_loss: {ce_out_models_loss.item():.4f}, adv_loss: {adv_loss.item():.4f}")
            
        return sample.detach()


    def find_target_labels_for_sample(self, sample: torch.Tensor, original_label: torch.Tensor, out_models: list[Union[int, torch.nn.Module]], device: Union[torch.device, str] = 'cpu'):
        target_labels = []
        for model_index, model in enumerate(out_models):
            if isinstance(model, torch.nn.Module):
                pass
            elif isinstance(model, int):
                model = self.shadow_manager.get_model(model)
            else:
                raise ValueError('Model should be either a torch Module or an integer referring to the id of the shadow model!')
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                logits = model(sample)
                pred_label = torch.argmax(logits, dim=1)
                if pred_label.item() != original_label.item():
                    target_labels.append(pred_label.item())
                else:
                    target_labels.append(np.argsort(logits.detach().cpu().numpy())[0][-2])
        target_labels = torch.tensor(target_labels, device=device)
        return target_labels




    # def relabel_shadow_dataset(self, train_config: TrainConfigs):
    #     # Relabel the shadow dataset according to the predictions of the target model, as done in the transfer attack of "Label-Only Membership Inference Attacks" (https://proceedings.mlr.press/v139/choquette-choo21a/choquette-choo21a.pdf).
    #     shadow_dataset = self.shadow_manager.get_dataset(index=0, labels='original')
    #     shadow_loader = torch.utils.data.DataLoader(shadow_dataset, batch_size=train_config.batch_size, shuffle=False)
    #     all_relabels = []
    #     device = self.get_device(train_config.device)
    #     self.defender_model.to(device)
    #     with torch.no_grad():
    #         for batch_index, (data, original_labels, _, _) in enumerate(shadow_loader):
    #             # print(f"original_labels shape: {original_labels.shape}")
    #             self.logger.print_it_same_line(f"DHAttack: relabelling batch {batch_index+1}/{len(shadow_loader)}...", console_only=True)
    #             data = data.to(device)
    #             preds = torch.argmax(self.defender_model(data), dim=1).cpu()
    #             all_relabels.append(preds)
    #     self.logger.set_logger_newline(console_only=True)
    #     all_relabels = torch.cat(all_relabels, dim=0)
    #     # print(f"all_relabels shape: {all_relabels.shape}, expected: ({len(shadow_dataset)},)")

    #     # Relabeling the shadow dataset with the obtained relabels
    #     distilled_dataset = copy.deepcopy(shadow_dataset)
    #     try:
    #         distilled_dataset.set_targets(all_relabels)
    #     except AttributeError:
    #         distilled_dataset.targets = all_relabels
    #     return distilled_dataset

    # def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
    #     if isinstance(device, str):
    #         device = self.get_device(dev_str=device)
    #     self.logger.print_it('DHAttack attacker: measuring attack effectiveness...')
    #     start = time.time()
    #     audit_dataset = self.audit_manager.get(labels='original')
    #     scores, decisions = self.infer_dataset(dataset=audit_dataset,
    #                                             device=device)
    #     stop = time.time()
    #     h, m, s = convert_to_hms(stop-start)
    #     self.logger.print_it('DHAttack attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
    #     metrics = self.compute_stats(scores)
    #     mia_audit_dataset = self.audit_manager.get(labels='mia')
    #     correct_decisions = (decisions == np.array([label for _, (_, label, _, _) in enumerate(mia_audit_dataset)]))
    #     attack_accuracy = np.mean(correct_decisions)
    #     self.logger.print_it(f"DHAttack attacker: attack accuracy at optimal threshold is {attack_accuracy:.4f}.")
    #     return metrics

    # @torch.no_grad()
    # def fixedbd_over_model(self, model: torch.nn.Module, inputs: torch.Tensor, true_labels: torch.Tensor) -> np.ndarray:
    #     device = inputs.device
    #     model.eval()
    #     x_fixed = self.construct_fixed_input(inputs).to(device)
    #     true_labels = true_labels.to(device)
    #     x_diff = x_fixed - inputs
    #     distances = torch.ones(inputs.shape[0], device=device) * self.attack_configs.n_queries
    #     mask = torch.ones(inputs.shape[0], device=device, dtype=torch.bool)
    #     for k in range(self.attack_configs.n_queries):
    #         x_masked = k/self.attack_configs.n_queries * x_diff + inputs
    #         # self.debug_image(x_masked[0])
    #         logits = model(x_masked)
    #         predictions = torch.argmax(logits, dim=1)
    #         predictions_to_check = torch.where(mask, predictions, -1)
    #         labels_to_check = torch.where(mask, true_labels, -1)
    #         # predictions_to_check = predictions[torch.where(mask == True)]
    #         # current_wrong_predictions = torch.where(predictions_to_check != true_labels[torch.where(mask == True)])
    #         current_wrong_predictions = torch.where(predictions_to_check != labels_to_check)
    #         distances[current_wrong_predictions] = k
    #         mask[current_wrong_predictions] = False
    #         if all(mask == False):
    #             break
    #     self.logger.print_it(f"DEBUG: Computed FixedDB distances for {inputs.shape[0]} samples in {k+1} queries per sample.")
    #     return distances.cpu().numpy()
    
    # @torch.no_grad()
    # def rel_score(self, inputs: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
    #     device = inputs.device
    #     shadow_models = self.shadow_manager.get_all_models()
    #     # print(f"shadow_models: {shadow_models}")
    #     dist_matrix = np.zeros((inputs.shape[0], len(shadow_models)))
    #     for i, shadow_model in shadow_models.items():
    #         self.logger.print_it_same_line(f"Computing FixedDB distance with shadow model {i+1}/{len(shadow_models)}...", console_only=True)
    #         shadow_model = shadow_model.to(device)
    #         shadow_model.eval()
    #         distances = self.fixedbd_over_model(model=shadow_model, inputs=inputs, true_labels=true_labels)
    #         dist_matrix[:, i] = distances
    #     self.logger.set_logger_newline(console_only=True)
    #     self.logger.print_it(f"Computing normal distribution parameters for REL scores based on shadow models...")
    #     shadow_means = np.mean(dist_matrix, axis=1)
    #     assert shadow_means.shape == (inputs.shape[0],)
    #     shadow_stds = np.std(dist_matrix, axis=1)
    #     assert shadow_stds.shape == (inputs.shape[0],)
    #     self.logger.print_it(f"Computing FixedDB distance with defender model...")
    #     target_model = self.defender_model.to(device)
    #     target_model.eval()
    #     target_distances = self.fixedbd_over_model(model=target_model, inputs=inputs, true_labels=true_labels)
    #     self.logger.print_it(f"Computing REL scores...")
    #     rel_scores = norm.cdf(target_distances, loc=shadow_means, scale=shadow_stds)
    #     return torch.asarray(rel_scores)

    # def construct_fixed_input(self, inputs: torch.Tensor) -> torch.Tensor:
    #     B, C, H, W = inputs.shape
    #     dataset_transformer = self.shadow_manager.get_dataset(index=0, labels='original').transform
    #     if self.attack_configs.fixed_input_mode == 'white':
    #         fixed_x_np = np.full((H, W, C), 255, dtype=np.uint8)  # H, W, C
    #     elif self.attack_configs.fixed_input_mode == 'black':
    #         fixed_x_np = np.full((H, W, C), 0, dtype=np.uint8)
    #     elif self.attack_configs.fixed_input_mode == 'random':
    #         rng = np.random.default_rng(seed=12345)
    #         fixed_x_np = rng.integers(0, 256, size=(H, W, C), dtype=np.uint8)
    #     else:
    #         raise ValueError(f"Unsupported fixed_input_mode {self.attack_configs.fixed_input_mode} for DHAttack! Supported modes are: 'white' and 'black'.")
    #     fixed_x_pil = Image.fromarray(fixed_x_np)
    #     x_fixed = dataset_transformer(fixed_x_pil)
    #     x_fixed = x_fixed.unsqueeze(0)
    #     x_fixed_batch = x_fixed.repeat(B, 1, 1, 1)
    #     return x_fixed_batch
    
    # def construct_random_samples(self, shape: tuple[int], n_samples: int) -> torch.Tensor:
    #     C, H, W = shape
    #     dataset_transformer = self.shadow_manager.get_dataset(index=0, labels='original').transform
    #     samples = []
    #     for _ in range(n_samples):
    #         random_x_np = np.random.randint(0, 256, (H, W, C), dtype=np.uint8)
    #         random_x_pil = Image.fromarray(random_x_np)
    #         random_x_torch = dataset_transformer(random_x_pil)
    #         samples.append(random_x_torch)
    #     return torch.stack(samples, dim=0)  # n_samples, C, H, W

    # # def find_optimal_threshold(self, device: Union[torch.device, str] = 'cpu'):
    # #     if isinstance(device, str):
    # #         device = self.get_device(dev_str=device)
    # #     n_synthetic_samples = 200
    # #     dummy_sample = self.shadow_manager.get_dataset(index=0, labels='original')[0][0]
    # #     synthetic_samples = self.construct_random_samples(shape=dummy_sample.shape, n_samples=n_synthetic_samples).to(device)
    # #     rel_scores_synthetic = self.rel_score(inputs=synthetic_samples, true_labels=torch.zeros(n_synthetic_samples, dtype=torch.long, device=device))
    # #     optimal_threshold = np.percentile(rel_scores_synthetic, q=98)
    # #     self.attack_threshold = optimal_threshold
    # #     self.logger.print_it(f"Optimal threshold found on synthetic samples: {optimal_threshold:.4f}.")

    # def find_optimal_threshold(self, device: Union[torch.device, str] = 'cpu'):
    #     self.attack_threshold = 0.01

    # @torch.no_grad()
    # def infer_dataset(self, dataset: torch.utils.data.Dataset, device: Union[torch.device, str] = 'cpu'):
    #     assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_dataset(...) for Transfer MIA."
    #     if isinstance(device, str):
    #         device = self.get_device(dev_str=device)
    #     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    #     all_scores = []
    #     for batch_id, (batch_x, batch_y, _, _) in enumerate(dataloader):
    #         self.logger.print_it_same_line(f"DHAttack attacker: processing batch {batch_id+1}/{len(dataloader)} for inference...", console_only=True)
    #         batch_x = batch_x.to(device)
    #         batch_y = batch_y.to(device)
    #         scores = self.rel_score(inputs=batch_x, true_labels=batch_y)
    #         all_scores.append(scores.cpu())
    #     self.logger.set_logger_newline(console_only=True)
    #     all_scores = torch.stack(all_scores).numpy()
    #     all_decisions = (all_scores > self.attack_threshold).astype(np.int64)
    #     return all_scores, all_decisions
    
    # @torch.no_grad()
    # def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
    #     assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_batch(...) for Transfer MIA."
    #     if isinstance(device, str):
    #         device = self.get_device(dev_str=device)
    #     model = model.to(device)
    #     model.eval()
    #     batch_x = batch_x.to(device)
    #     batch_y = batch_y.to(device)
    #     scores = self.rel_score(inputs=batch_x, true_labels=batch_y).numpy()
    #     decisions = (scores > self.attack_threshold).astype(np.int64)
    #     return scores.cpu().numpy(), decisions

    # @torch.no_grad()
    # def infer_single(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
    #     assert self.attack_threshold is not None, "Call find_optimal_threshold(...) before infer_single(...) for Transfer MIA."
    #     if isinstance(device, str):
    #         device = self.get_device(dev_str=device)
    #     model = model.to(device)
    #     model.eval()
    #     x = x.unsqueeze(0).to(device)
    #     y = y.unsqueeze(0).to(device)
    #     score = self.rel_score(inputs=x, true_labels=y).numpy()
    #     decision = (score > self.attack_threshold).astype(np.int64)
    #     return score.item(), decision.item()
    
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

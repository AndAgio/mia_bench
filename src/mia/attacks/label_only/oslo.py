
from typing import Union
import time
import math
import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy
from src.mia.attacks.base_mia import BaseMIA
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, TrainConfigs
from src.utils import convert_to_hms



class OsloMIA(BaseMIA):
    # Implementation of "OSLO: One-Shot Label-Only Membership Inference Attacks" (https://proceedings.neurips.cc/paper_files/paper/2024/file/71f88122d414cfeb455ac0ed932fbe1f-Paper-Conference.pdf).
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Oslo MIA!")
        assert self.shadow_configs.mode == 'offline', f"Oslo MIA attacker should be used with offline shadow models, but found mode={self.shadow_configs.mode} instead!"
        if self.shadow_configs.n_shadow_datasets != 1:
            self.logger.print_it(f"Oslo MIA attacker [WARNING]: when using oslo MIA, only 1 shadow dataset must be used! Modifying shadow_configs on the fly to set n_shadow_datasets to 1.")
            self.shadow_configs.n_shadow_datasets = 1
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Oslo MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.shadow_manager.add_dataset_replicas(index=0, 
                                                n_replicas=self.attack_configs.n_models)
        self.logger.print_it('Oslo MIA attacker: definition of boundary model...')
        self.shadow_manager.build_shadow_models(n_models=self.attack_configs.n_models,
                                                model_configs=self.model_configs,
                                                same_model_arch=self.attack_configs.same_arch)

    def optimize(self, train_config: TrainConfigs):
        self.logger.print_it('Oslo MIA attacker: training all shadow models. This will take a while. Sit back and chill...')
        start = time.time()
        self.shadow_manager.train_all(train_configs=train_config,
                                        labels_mode='original')
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Oslo MIA attacker: Done optimizing shadow models. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        self.build_ensembles()

    def build_ensembles(self):
        self.logger.print_it('Oslo MIA attacker: building ensemble models for transferable adversarial attack generation...')
        start = time.time()
        n_source_models = math.floor(self.shadow_manager.get_n_models() * self.attack_configs.source_models_ratio)
        n_val_models = self.shadow_manager.get_n_models() - n_source_models
        self.logger.print_it(f"Oslo MIA attacker: using {n_source_models} shadow models as source models to build the ensemble, and {n_val_models} shadow models as validation.")
        self.source_models_ensemble = MultiEnsemble(model_list=[self.shadow_manager.get_model(index=i) for i in range(n_source_models)])
        self.val_models_ensemble = MultiEnsemble(model_list=[self.shadow_manager.get_model(index=i) for i in range(n_source_models, self.shadow_manager.get_n_models())])
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Oslo MIA attacker: Done building ensemble models. It took {}:{:02d}:{:02d}...'.format(h, m, s))


    def measure_effectiveness(self, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        self.logger.print_it('Supervised Boundary MIA attacker: measuring attack effectiveness...')
        start = time.time()
        audit_dataset = self.audit_manager.get(labels='original')
        scores, decisions = self.infer_dataset(model=self.defender_model.to(device),
                                                dataset=audit_dataset,
                                                device=device)
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Supervised Boundary MIA attacker: Done measuring attack effectiveness. It took {}:{:02d}:{:02d}...'.format(h, m, s))
        
        metrics = self.compute_stats(scores, decisions=decisions)
        return metrics

    @torch.no_grad()
    def label_pred(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """
        Returns hard labels (argmax) for a batch x.
        """
        model.eval()
        logits = model(inputs)
        return torch.argmax(logits, dim=1)

    def infer_dataset(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        all_scores = []
        for batch_id, (batch_x, batch_y, _, _) in enumerate(dataloader):
            self.logger.print_it_same_line(f"Boundary Distance MIA attacker: processing batch {batch_id+1}/{len(dataloader)} for inference. This may take a while...", console_only=True)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            scores = self.infer_batch(model=model, batch_x=batch_x, batch_y=batch_y, device=device)
            all_scores.append(scores)
        self.logger.set_logger_newline(console_only=True)
        all_scores = torch.cat(all_scores, dim=0).numpy()
        all_decisions = (all_scores >= self.attack_configs.threshold).astype(np.int64)
        return all_scores, all_decisions

    def infer_batch(self, model: torch.nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        if isinstance(device, str):
            device = self.get_device(dev_str=device)
        model = model.to(device)
        model.eval()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        _, score = self._score(inputs=batch_x, targets=batch_y)
        return score.detach().cpu()

    def _score(self, inputs: torch.Tensor, targets: torch.Tensor):
        # Compute the attack score for a batch of samples.
        loss_fn = torch.nn.CrossEntropyLoss()

        batch_size = inputs.shape[0]
        g = torch.zeros_like(inputs)
        delta = torch.zeros_like(inputs)
        self.source_models_ensemble = self.source_models_ensemble.to(inputs.device)
        self.val_models_ensemble = self.val_models_ensemble.to(inputs.device)
        eps_list = [(idx + 1) * (self.attack_configs.max_epsilon / self.attack_configs.K) for idx in range(self.attack_configs.K)]
        mask = torch.ones((batch_size,), dtype=torch.bool, device=inputs.device)

        for eps in eps_list:
            eps /= 255.0
            step_size = 1.25 * eps / self.attack_configs.N
            delta = torch.autograd.Variable(delta.data, requires_grad=True)

            for _ in range(self.attack_configs.N):
                delta.requires_grad_()
                adv = inputs + delta
                adv = torch.clamp(adv, 0, 1)
                with torch.enable_grad():
                    ensem_logits = self.source_models_ensemble(adv)
                    loss = loss_fn(ensem_logits, targets)

                PGD_grad = torch.autograd.grad(loss.sum(), [delta])[0].detach()

                if self.attack_configs.ga_mode == 'difgsm':
                    g[mask] = PGD_grad[mask].clone()
                elif self.attack_configs.ga_mode == 'mifgsm':
                    PGD_noise = PGD_grad / torch.abs(PGD_grad).mean(dim=(1, 2, 3), keepdim=True)
                    MOMENTUM = 1
                    g[mask] = g[mask] * MOMENTUM + PGD_noise[mask]
                elif self.attack_configs.ga_mode == 'tifgsm':
                    KERNEL_SIZE = 5
                    kernel = get_kernel(KERNEL_SIZE).to(device=PGD_grad.device, dtype=PGD_grad.dtype)
                    PGD_grad = torch.nn.functional.conv2d(PGD_grad, weight=kernel, stride=(1, 1), groups=3, padding=(KERNEL_SIZE - 1) // 2)
                    g[mask] = PGD_grad[mask].clone()
                elif self.attack_configs.ga_mode == 'tmifgsm':
                    KERNEL_SIZE = 5
                    MOMENTUM = 1
                    kernel = get_kernel(KERNEL_SIZE).to(device=PGD_grad.device, dtype=PGD_grad.dtype)
                    PGD_grad = torch.nn.functional.conv2d(PGD_grad, weight=kernel, stride=(1, 1), groups=3, padding=(KERNEL_SIZE - 1) // 2)
                    PGD_noise = PGD_grad / torch.abs(PGD_grad).mean(dim=(1, 2, 3), keepdim=True)
                    g[mask] = g[mask] * MOMENTUM + PGD_noise[mask]
                else:
                    raise ValueError(f"Unsupported gradient-based adversarial mode '{self.attack_configs.ga_mode}' for OSLO MIA.")
                delta = torch.autograd.Variable(delta.data + step_size * torch.sign(g), requires_grad=True)
                delta = torch.autograd.Variable(torch.clamp(delta.data, -eps, eps), requires_grad=True)

            g.zero_()
            with torch.no_grad():
                tmp = inputs + delta
                tmp = torch.clamp(tmp, 0, 1)
                output = self.val_models_ensemble(tmp).detach()
            prob = torch.nn.functional.softmax(output, dim=1)
            conf = prob[torch.arange(batch_size, device=prob.device), targets.long()]
            mask = (conf >= self.attack_configs.threshold) # it keeps perturbing the ones with confidence above the threshold, and stops perturbing the ones that are already below the threshold (early stopping)

            # early stopping
            if mask.sum() == 0:
                break

        adv_inputs = torch.autograd.Variable(inputs + delta, requires_grad=False)
        adv_inputs = torch.autograd.Variable(torch.clamp(adv_inputs, 0, 1), requires_grad=False)
        dist = torch.abs(adv_inputs - inputs).reshape(batch_size, -1).max(dim = -1)[0]
        return adv_inputs, dist



class JITModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = torch.compile(model)

    def forward(self, x):
        return self.model(x)


class MultiEnsemble(torch.nn.Module):
    def __init__(self, model_list: list[torch.nn.Module]):
        super(MultiEnsemble, self).__init__()
        self.model_list = model_list
        self.length = len(self.model_list)

    def forward(self, x: torch.Tensor):
        output = torch.cat([self.model_list[idx](x).unsqueeze(1) for idx in range(self.length)], dim = 1)
        return output.mean(dim = 1)


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = scipy.stats.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def get_kernel(kernel_size=7):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3).transpose(2, 3, 0, 1)
    stack_kernel = torch.from_numpy(stack_kernel)
    return stack_kernel

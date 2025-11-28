import os
import pathlib
import sys
import math
import time
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from src.data.multi import MultiDatasets
from src.models import get_model
from src.optimizers import SAM, SGD, Adam, ESAM, WSAM, LookSAM, FriendlySAM
from src.optimizers.utils import enable_running_stats, disable_running_stats
from src.optimizers.schedulers import GradualWarmupScheduler, CosineAnnealingWarmupRestarts
from src.utils.log import get_logger
from src.utils import convert_to_hms
from .utils import TrainCheckpoint, TrainStats, TrainConfigs, EpochStats

from src.utils.variables import DEFAULT_LOG_FOLDER


class TrainManager():
    def __init__(self, train_configs: TrainConfigs, name: str,
                log_folder: pathlib.Path = DEFAULT_LOG_FOLDER, logging_mode: str = 'smart',):
        self.train_configs = train_configs
        self.name = name
        self.logger = get_logger(name=name, log_folder=log_folder, mode=logging_mode)

        self.setup_folders(train_configs=self.train_configs)
        self.set_devices_and_seed(train_configs=self.train_configs)
        self.reset_running_stats()

    
    def reset_config(self, configs: TrainConfigs):
        self.train_configs = configs
        self.setup_folders(train_configs=self.train_configs)
        self.set_devices_and_seed(train_configs=self.train_configs)
        self.reset_running_stats()
    

    def setup_folders(self, train_configs: TrainConfigs):
        models_folder = os.path.join(train_configs.ckpts_folder, self.name, str(train_configs.seed))
        os.makedirs(models_folder, exist_ok=True)
        self.models_folder = models_folder
        self.ckpts_folder = train_configs.ckpts_folder
        resume_folder = os.path.join(train_configs.resume_ckpts_folder, self.name, str(train_configs.seed))
        os.makedirs(resume_folder, exist_ok=True)
        self.resume_folder = resume_folder


    def set_devices_and_seed(self, train_configs: TrainConfigs):
        self._setup_distributed_training(distributed=train_configs.distributed,
                                        device=train_configs.device)
        self._setup_device(distributed=train_configs.distributed,
                            device=train_configs.device)
        self._setup_seed(seed=train_configs.seed)


    def _setup_distributed_training(self, distributed: bool = False, device: str = 'cpu'):
        if distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
        else:
            self.local_rank = device if torch.cuda.is_available() and device != 'cpu' else 'mps' if torch.backends.mps.is_available() and device != 'cpu' else 'cpu'
            self.global_rank = device if torch.cuda.is_available() and device != 'cpu' else 'mps' if torch.backends.mps.is_available() and device != 'cpu' else 'cpu'
        self.distributed = distributed


    def _setup_device(self, distributed: bool = False, device: str = 'cpu'):
        # Set appropriate devices
        if distributed:
            dev_str = 'cuda:{}'.format(self.local_rank)
            self.device = torch.device(dev_str)
        else:
            if torch.cuda.is_available() and device != 'cpu':
                dev_str = 'cuda:{}'.format(device)
                self.device = torch.device(dev_str)
                self.logger.print_it('Runner is setup using CUDA enabled device: {}'.format(torch.cuda.get_device_name(dev_str)))
            elif torch.backends.mps.is_available() and device != 'cpu':
                self.logger.print_it('Found MacBook with M# device. I will try to use it for training...')
                dev_str = 'mps'
                self.device = torch.device(dev_str)
                self.logger.print_it('Runner is setup using MPS enabled device: {}'.format(self.device))
            else:
                self.device = torch.device('cpu')
                self.logger.print_it('Runner is setup using CPU! Training and inference will be very slow!')
            cudnn.benchmark = True  # Should make training go faster for large models


    def _setup_seed(self, seed: int = 12345):
        # Set random seed for initialization
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed


    def save_model(self, name):
        torch.save(self.model, os.path.join(self.models_folder, name))


    def load_best_model(self):
        self.model = torch.load(os.path.join(self.models_folder, 'best.pt'))
        self.model = self.model.to(self.device)


    def load_last_model(self):
        self.model = torch.load(os.path.join(self.models_folder, 'last.pt'))
        self.model = self.model.to(self.device)


    def setup_model_from_name(self, model_name: str, dataset_info: dict):
        self.logger.print_it('Setting up model "{}"...'.format(model_name))
        # Setup model
        model = get_model(model_name=model_name, dataset_info=dataset_info)
        # Move model to device
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = model.to(self.device)
        self.model_name = model_name
        self.logger.print_it('Model setup done!')

    def set_model(self, model: nn.Module):
        self.logger.print_it('Setting up model "{}"...'.format(model.name))
        self.model = copy.deepcopy(model)
        # Move model to device
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = model.to(self.device)
        self.model_name = model.name
        self.logger.print_it('Model setup done!')

    def get_model_name(self):
        return self.model_name

    def setup_loss(self, loss: str):
        self.logger.print_it('Setting up {} loss...'.format(loss))
        if loss == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        else:
            print('Specified loss "{}" not recognized!'.format(loss))

    def setup_optimizer(self, optimizer: str, lr: float = None):
        self.logger.print_it('Setting up "{}" optimizer...'.format(optimizer))
        if optimizer == 'adam':
            self.optimizer = Adam(params=self.model.parameters(),
                                lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = SGD(params=self.model.parameters(), 
                                lr=lr,
                                momentum=0.9,
                                nesterov=False,
                                weight_decay=0.0005)
        elif optimizer.split('_')[-1] == 'sam':
            adaptive = True if optimizer.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = SAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=lr,
                                rho=0.05,
                                adaptive=adaptive,
                                momentum=0.9,
                                nesterov=False,
                                weight_decay=0.0005)
        elif optimizer.split('_')[-1] == 'esam':
            adaptive = True if optimizer.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = ESAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=lr,
                                rho=0.05,
                                beta=1,
                                gamma=0.5,
                                adaptive=adaptive,
                                momentum=0.9,
                                nesterov=False,
                                weight_decay=0.0005)
        elif optimizer.split('_')[-1] == 'wsam':
            adaptive = True if optimizer.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = WSAM(params=self.model.parameters(),
                                base_optimizer=SGD,
                                lr=lr,
                                rho=0.05,
                                gamma=0.9,
                                sam_eps=1e-12,
                                adaptive=adaptive,
                                decouple=True,
                                max_norm=None,
                                momentum=0.9,
                                nesterov=False,
                                weight_decay=0.0005)
        elif optimizer.split('_')[-1] == 'looksam':
            adaptive = True if optimizer.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = LookSAM(params=self.model.parameters(),
                                    base_optimizer=SGD,
                                    rho=0.05,
                                    k=10,
                                    alpha=0.7,
                                    adaptive=adaptive,
                                    use_gc=False,
                                    perturb_eps=1e-12,
                                    nesterov=False,
                                    weight_decay=0.0005)
        elif optimizer.split('_')[-1] == 'friendlysam':
            adaptive = True if optimizer.split('_')[0] in ['a', 'ad', 'ada', 'adap', 'adaptive'] else False
            self.optimizer = FriendlySAM(params=self.model.parameters(),
                                        base_optimizer=SGD,
                                        rho=0.05,
                                        sigma=1,
                                        lmbda=0.9,
                                        adaptive=adaptive,
                                        perturb_eps=1e-12,
                                        momentum=0.9,
                                        nesterov=False,
                                        weight_decay=0.0005)
        else:
            raise ValueError('Specified optimizer "{}" not supported. Options are: adam and sgd and sam'.format(optimizer))

    def setup_lr_scheduler(self, lr_sched: str, epochs: int = None, lr: float = None):
        self.logger.print_it('Setting up "{}" learning rate scheduler...'.format(lr_sched))
        if lr_sched == 'const':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1)
        elif lr_sched == 'warmup_step':
            assert epochs is not None and epochs > 0
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(epochs/3), gamma=0.1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif lr_sched == 'warmup_exp':
            assert epochs is not None and epochs > 0
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif lr_sched == 'warmup_cosine':
            assert epochs is not None and epochs > 0
            assert lr is not None and 0 < lr < 1
            cycle_steps = math.ceil(epochs/5)
            warmup_steps = math.ceil(cycle_steps/10)
            max_lr=self.settings.lr
            min_lr=max_lr/100
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=cycle_steps, cycle_mult=1.0, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, gamma=0.5)
        elif lr_sched == 'step':
            assert epochs is not None and epochs > 0
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(epochs/3), gamma=0.1)
        elif lr_sched == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        elif lr_sched == 'cosine':
            assert epochs is not None and epochs > 0
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        else:
            raise ValueError('Learning rate scheduler "{}" not available!'.format(lr_sched))

    def setup_training(self, 
                        loss: str,
                        optimizer: str,
                        lr: float,
                        lr_sched: str,
                        epochs: int):
        self.logger.print_it('Setting up training...')
        self.setup_loss(loss=loss)
        self.setup_optimizer(optimizer=optimizer,
                            lr=lr,)
        self.setup_lr_scheduler(lr_sched=lr_sched,
                                epochs=epochs, 
                                lr=lr)
        self.logger.print_it('Training setup done!')


    def setup_dataloaders(self, dataset: MultiDatasets, batch_size: int = 128):
        try:
            train_dataset = dataset.get('train')
            self.run_train = True
        except:
            raise ValueError('At least the train split should be in the dataset!')
        try:
            test_dataset = dataset.get('test')
            self.run_test = True
        except:
            self.run_test = False
        if self.distributed:
            if self.run_train:
                self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                                pin_memory=True, shuffle=False,
                                                sampler=DistributedSampler(train_dataset))
            if self.run_test:
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                                pin_memory=True, shuffle=False,
                                                sampler=DistributedSampler(test_dataset))
        else:
            if self.run_train:
                self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            if self.run_test:
                self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    def initialize_train(self, 
                        dataset: MultiDatasets,
                        model: str | nn.Module,
                        configs: TrainConfigs,
                        ):
        self.logger.print_it('Initializing training...')

        if not configs.__eq__(self.train_configs):
            self.logger.print_it('Found different training configurations in the initialize_train method. Resetting the trainer configs...')
            self.reset_configs(configs)

        self.setup_dataloaders(dataset=dataset,
                                batch_size=self.train_configs.batch_size)
        if isinstance(model, nn.Module):
            self.set_model(model)
        elif isinstance(model, str):
            self.setup_model_from_name(model_name=model, dataset_info=dataset.get_info())
        else:
            raise ValueError('Not recognizing model given!')
        self.setup_training(loss=self.train_configs.loss,
                            optimizer=self.train_configs.optimizer,
                            lr=self.train_configs.lr,
                            lr_sched=self.train_configs.lr_sched,
                            epochs=self.train_configs.epochs)
        self.reset_running_stats()

        self.logger.print_it('Training initialization completed!')


    def reset_running_stats(self):
        self.running_stats = TrainStats()

    
    def reset_epoch_stats(self):
        self.epoch_stats = EpochStats()
    

    def train(self, return_model: bool = True, return_stats: bool = False):
        while(self.running_stats.epoch <= self.train_configs.epochs):
            start_time = time.time()
            # Train and test (if possible) for one epoch
            train_acc = self.train_epoch()
            # self.running_stats['train_accs'].append(train_acc)
            if self.run_test:
                test_acc = self.test_epoch()
                # self.running_stats['test_accs'].append(test_acc)

            epoch_time = time.time() - start_time
            self.running_stats.update_time(epoch_time)
            # self.elapsed_time += epoch_time
            h, m, s = convert_to_hms(self.running_stats.elapsed_time)
            self.logger.print_it('Elapsed time for epoch {}: {}:{:02d}:{:02d}'.format(self.running_stats.epoch,h,m,s))

            self.scheduler.step()

            self.save_model('epoch_{}.pt'.format(self.running_stats.epoch))

            # Save checkpoint when best model
            if self.running_stats.is_new_best(test_acc, mode='test' if self.run_test else 'train'):
                    self.logger.print_it('New Best model on {} at epoch {}: \t Top1-acc = {:.2f}'.format('test' if self.run_test else 'train',
                                                                                                        self.running_stats.epoch, 
                                                                                                        test_acc*100 if self.run_test else train_acc*100 ))
                    self.save_model('best.pt')
            
            # Update history of accuracies
            self.running_stats.update_train_accs(train_acc)
            self.running_stats.update_test_accs(test_acc)

            
            # Save model when last epoch
            if self.running_stats.epoch == self.train_configs.epochs:
                self.logger.print_it('Saving last model...')
                self.save_model('last.pt')

            if self.local_rank == 0 or not self.distributed:
                self.store_resume_ckpt()
                # if self.settings.dataset in ['imagenet']:
                #     message = 'EXPERIMENT: {}\nEPOCH: {}/{}\nLAST ACC = {}\nBEST ACC = {}'.format(self.experiment_name, self.epoch, self.settings.epochs, test_acc, self.best_acc)
                #     send_update_via_telegram(message)

            self.running_stats.increase_epoch()

        h, m, s = convert_to_hms(self.running_stats.elapsed_time)
        self.logger.print_it('Training for "{}" with seed {} completed in: {}:{:02d}:{:02d}'.format(self.name, self.seed, h, m, s))

        # message = 'EXPERIMENT: {}\nTraining of "{}" for "{}" completed in: {}:{:02d}:{:02d}\nLAST ACC = {}\nBEST ACC = {}'.format(self.experiment_name, self.settings.model, self.settings.dataset, h, m, s, test_acc, self.best_acc)
        # send_update_via_telegram(message)

        if return_model:
            if return_stats:
                return self.model, self.running_stats
            else:
                return self.model
        else:
            if return_stats:
                return self.running_stats
        return


    def train_epoch(self):
        self.reset_epoch_stats()

        self.model.train()

        if self.distributed:
            self.train_loader.sampler.set_epoch(self.running_stats.epoch)
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            self.train_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.train_loader))

        self.logger.set_logger_newline()
        # Add test accuracy to dict
        acc = self.epoch_stats.avg_acc('train')
        return acc

    def train_step(self, inputs, targets, batch_idx=0, total_batches=0):
        # Map to available device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Compute loss and predictions
        if type(self.optimizer) in [SAM, ESAM, WSAM, LookSAM, FriendlySAM]:
            # Working with closure
            def closure(inputs, targets, mean=True, backward=True, run_stats=True):
                if run_stats:
                    enable_running_stats(self.model)
                else:
                    disable_running_stats(self.model)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                if mean:
                    loss = loss.mean()
                if backward:
                    loss.backward()
                return loss, outputs
            self.optimizer.step(closure, inputs, targets)
            self.optimizer.zero_grad()
            loss, outputs = self.optimizer.get_first_closure_outputs()
            # An alternative to running with closure is to run manually both steps of the sam-like optimizers, like the following.
            # # first forward-backward step
            # enable_running_stats(self.model)
            # outputs = self.model(inputs)
            # loss = self.criterion(outputs, targets).mean()
            # loss.backward()
            # self.optimizer.first_step(zero_grad=True)
            # # second forward-backward step
            # disable_running_stats(self.model)
            # self.criterion(self.model(inputs), targets).mean().backward()
            # self.optimizer.second_step(zero_grad=True)
        else:
            # Forward propagation, compute loss, get predictions
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)        
        self.epoch_stats.increase('train_loss', loss.item())
        self.epoch_stats.increase('train_total', targets.size(0))
        self.epoch_stats.increase('train_correct', predicted.eq(targets.data).cpu().sum().item())

        # Print message on console
        metrics = {'loss': self.epoch_stats.avg_loss('train'),
                    'acc': self.epoch_stats.avg_acc('train')}
        self.print_message(epoch_index=self.running_stats.epoch, total_epochs=self.train_configs.epochs,
                        index_batch=batch_idx+1, total_batches=total_batches,
                        metrics=metrics, mode='train')
    
    def test_epoch(self):
        self.model.eval()
        if self.distributed:
            self.test_loader.sampler.set_epoch(self.running_stats.epoch)
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            self.test_step(inputs, targets, batch_idx=batch_idx, total_batches=len(self.test_loader))
        self.logger.set_logger_newline()
        # Add test accuracy to dict
        acc = self.epoch_stats.avg_acc('test')
        return acc
    
    def test_step(self, inputs, targets, batch_idx=0, total_batches=0):
        # Map to available device
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        # Forward propagation, compute loss, get predictions
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss = loss.mean()
        self.epoch_stats.increase('test_loss', loss.item())
        _, predicted = torch.max(outputs.data, 1)
        self.epoch_stats.increase('test_total', targets.size(0))
        self.epoch_stats.increase('test_correct', predicted.eq(targets.data).cpu().sum().item())

        # Print message on console
        metrics = {'loss': self.epoch_stats.avg_loss('test'),
                    'acc': self.epoch_stats.avg_acc('test')}
        self.print_message(epoch_index=self.running_stats.epoch, total_epochs=self.train_configs.epochs,
                        index_batch=batch_idx+1, total_batches=total_batches,
                        metrics=metrics, mode='test')
    
    def store_resume_ckpt(self):
        state = TrainCheckpoint(train_stats=self.running_stats,
                                model_state=self.model.state_dict(),
                                opt_state=self.optimizer.state_dict(),
                                sched_state=self.scheduler.state_dict())
        torch.save(state, os.path.join(self.resume_folder, 'epoch={}.pth.tar'.format(self.running_stats.epoch)))

    def load_last_resume_ckpt(self):
        # TODO: Find optimal way to implement resume.
        # Issue URL: https://github.com/AndAgio/mia_bench/issues/4
        # assignee: AndAgio
        raise NotImplementedError('Still to be implemented!')
        checkpoint_folder = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        checkpoints = glob.glob(os.path.join(checkpoint_folder, '*.pth.tar'))
        found_epochs = [int(check.split('epoch=')[-1].split('.pth.tar')[0]) for check in checkpoints]
        if found_epochs == []:
            self.logger.print_it('No resume checkpoint found! Are you sure you wanted to resume?')
            self.initialize_train()
        else:
            self.initialize_train()
            last_epoch = max(found_epochs)
            checkpoint_to_load = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed), 'epoch={}.pth.tar'.format(last_epoch))
            # Read variables
            checkpoint = torch.load(checkpoint_to_load, map_location=self.device)
            self.epoch = checkpoint.epoch + 1
            self.best_acc = checkpoint['best_acc']
            self.elapsed_time = checkpoint['elapsed_time']
            self.test_accs = checkpoint['test_accs']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def print_message(self, epoch_index, total_epochs, index_batch, total_batches, metrics, mode='train'):
        if self.global_rank == 'cpu':
            message = 'CPU | EPOCH: {}/{} |'.format(epoch_index, total_epochs)
        elif self.global_rank == 'mps':
            message = 'MPS | EPOCH: {}/{} |'.format(epoch_index, total_epochs)
        else:
            message = 'GPU-{} | EPOCH: {}/{} |'.format(self.global_rank, epoch_index, total_epochs)
        bar_length = 10
        progress = float(index_batch) / float(total_batches)
        if progress >= 1.:
            progress = 1
        block = int(round(bar_length * progress))
        message += '[{}]'.format('=' * block + ' ' * (bar_length - block))
        message += '| {}: '.format(mode.upper())
        if metrics is not None:
            train_metrics_message = ''
            index = 0
            for metric_name, metric_value in metrics.items():
                train_metrics_message += '{}={:.5f}{} '.format(metric_name, metric_value,
                                                            ',' if index < len(metrics.keys()) - 1 else '')
                index += 1
            message += train_metrics_message
        message += '|'
        self.logger.print_it_same_line(message)

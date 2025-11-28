import os
import math
import numpy as np
import torch
import torch.nn as nn
import time
import pickle
import glob
from torchview import draw_graph
# Import custom stuff
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as T
from src.datasets import AircraftDataset, CIFAR100, CarsDataset, CubDataset, Fashion
from src.models import enable_running_stats, disable_running_stats
from src.models.base import get_base_model, get_base_model_features
from src.models.no_ski import NoSki, NoskiLoss
from src.models.no_ski_l0_only import NoSkiL0Only, NoSkiL0OnlyLoss
from src.models.hrn import HRN, TreeLoss
from src.models.multiplexnet import MultiPlexLoss, MultiPlexNet, logic_terms
from src.models.chmcnn import CHMCNN, ChmcnnLoss
from src.models.spl import SPL, SplLoss
from src.models.semloss import SemlossModel, HierarchySemanticLoss
from src.optimizers import SAM, SGD, Adam
from src.optimizers.schedulers import GradualWarmupScheduler, CosineAnnealingWarmupRestarts
from src.metrics.performance import Auprc, Jaccard, LevelsAcc
from src.metrics.flatness import HessianTraceCurvature, EpsilonFlatness
from .runner import Runner


class Trainer(Runner):
    def __init__(self, settings):
        super().__init__(settings=settings, experiment_name='train_{}_with_{}+{}_over_{}'.format(settings.model, settings.base_model, settings.base_model_pretrain_mode, settings.dataset))
        self.gather_dataset()
        self.setup_model()
        self.setup_training()

    def gather_dataset(self):
        self.logger.print_it('Gathering dataset "{}". This may take a while...'.format(self.settings.dataset))
        # Load the appropriate train and test datasets
        chosen_dataset = self.settings.dataset
        # Data transforms (normalization & data augmentation)
        if chosen_dataset in ['std_cars', 'aircrafts', 'cub_birds']:
            self.img_size = (448, 448)
            train_trans = T.Compose([T.Resize((550, 550)),
                                    T.RandomCrop(self.img_size, padding=8),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
            test_trans = T.Compose([T.Resize((550, 550)), T.CenterCrop(self.img_size), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif chosen_dataset in ['fashion']:
            self.img_size = (448, 448)
            train_trans = T.Compose([T.Resize((550, 550)),
                                    T.RandomCrop(self.img_size, padding=8),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True)])
            test_trans = T.Compose([T.Resize((550, 550)), T.CenterCrop(self.img_size), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        elif chosen_dataset in ['cifar']:
            self.img_size = (256, 256)
            train_trans = T.Compose([T.RandomCrop(32, padding=4, padding_mode='reflect'), 
                                    T.RandomHorizontalFlip(), 
                                    T.RandomRotation(degrees=(0, 45)),
                                    T.RandomResizedCrop(self.img_size, scale=(0.5, 0.9), ratio=(1, 1)), 
                                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                    T.ToTensor(), 
                                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True)])
            test_trans = T.Compose([T.Resize(self.img_size), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            raise ValueError('Dataset "{}" is not available!'.format(chosen_dataset))
        # Load dataset
        if chosen_dataset == 'std_cars':
            train_data = CarsDataset("data/std_cars", train=True, transform=train_trans, download=True)
            test_data = CarsDataset("data/std_cars", train=False, transform=test_trans, download=True)
        elif chosen_dataset == 'aircrafts':
            train_data = AircraftDataset("data/aircrafts", train=True, transform=train_trans, download=True)
            test_data = AircraftDataset("data/aircrafts", train=False, transform=test_trans, download=True)
        elif chosen_dataset == 'cub_birds':
            train_data = CubDataset("data/cub_birds", train=True, transform=train_trans, download=True)
            test_data = CubDataset("data/cub_birds", train=False, transform=test_trans, download=True)
        elif chosen_dataset == 'cifar':
            train_data = CIFAR100('data/cifar100', train=True, transform=train_trans, download=True)
            test_data = CIFAR100('data/cifar100', train=False, transform=test_trans, download=True)
        elif chosen_dataset == 'fashion':
            train_data = Fashion("data/fashion", train=True, transform=train_trans, download=True)
            test_data = Fashion("data/fashion", train=False, transform=test_trans, download=True)
        else:
            raise ValueError('Dataset "{}" is not available!'.format(chosen_dataset))
        # Get dependency tree and adj matrix
        self.dependency_tree = train_data.get_class_dependency_tree()
        self.adj_matrix = train_data.get_adjecency_matrix()
        self.R = train_data.get_R()
        self.dimensions_to_eval = train_data.to_eval
        self.n_levels = train_data.get_n_levels()
        self.labels_levels = train_data.get_labels_levels()
        self.level_0_classes = train_data.get_level_0_classes()
        self.level_1_classes = train_data.get_level_1_classes()
        self.level_0_level_1_mapping = train_data.get_level_0_level_1_mapping()
        self.total_classes = train_data.get_n_total_classes()
        # Generate loaders
        train_length = train_data.__len__()  # Length training dataset
        train_indices = np.arange(train_length)  # Create arange
        np.random.shuffle(train_indices)  # Randomly Suffle training indices
        # self.train_loader = DataLoader(train_data,
        #                             batch_size=self.settings.batch_size,
        #                             sampler=SubsetRandomSampler(train_indices[: int(train_length * 0.8)]),)
        # self.val_loader = DataLoader(train_data,
        #                             batch_size=self.settings.batch_size,
        #                             sampler=SubsetRandomSampler(train_indices[int(train_length * 0.8) :]),)
        # self.test_loader = DataLoader(test_data, batch_size=self.settings.batch_size)
        self.train_loader = DataLoader(train_data, batch_size=self.settings.batch_size, shuffle=True)
        self.val_loader = DataLoader(test_data, batch_size=self.settings.batch_size, shuffle=False)
        # Disable Data Augmentation on Validation Set
        self.val_loader.dataset.transform = test_trans
        self.logger.print_it('Gathered dataset "{}" with {} samples in training set and {} samples in the test/validation set'.format(self.settings.dataset, len(train_data), len(test_data)))

    def setup_model(self):
        self.logger.print_it('Setting up model "{}"...'.format(self.settings.model))
        base_model = get_base_model(model=self.settings.base_model, mode=self.settings.base_model_pretrain_mode)
        base_model_features = get_base_model_features(model=self.settings.base_model)
        # model_out_features = 
        if self.settings.model == 'hrn':
            model = HRN(base_model=base_model, base_model_features=base_model_features,  levels_labels=self.labels_levels)
            self.criterion = TreeLoss(hierarchy=self.dependency_tree)               
        elif self.settings.model == 'multiplexnet':
            terms = logic_terms(level_0_classes=self.level_0_classes,
                                level_1_classes=self.level_1_classes,
                                level_0_level_1_mapping=self.level_0_level_1_mapping)
            model = MultiPlexNet(base_model=base_model, base_model_features=base_model_features, n_classes=len(self.level_0_classes), terms=terms)
            self.criterion = MultiPlexLoss(terms=terms, classes=self.level_0_classes)
        elif self.settings.model == 'chmcnn':
            model = CHMCNN(base_model=base_model, base_model_features=base_model_features, R=self.R, levels_labels=self.labels_levels, n_classes=self.total_classes, constrained_layer=True)
            self.criterion = ChmcnnLoss(R=self.R, hierarchy=self.dependency_tree, to_eval=torch.tensor(self.dimensions_to_eval))
        elif self.settings.model == 'spl':
            model = SPL(base_model=base_model, base_model_features=base_model_features, R=self.R, inputs_shape=torch.randn((self.settings.batch_size, 3, self.img_size[0], self.img_size[1])).shape, levels_labels=self.labels_levels, n_classes=self.total_classes, constrained_layer=True, dataset_name=self.settings.dataset) 
            self.criterion = SplLoss(cmpe=model.cmpe, hierarchy=self.dependency_tree)
        elif self.settings.model == 'sem_loss':
            model = SemlossModel(base_model=base_model, base_model_features=base_model_features, levels_labels=self.labels_levels, n_classes=self.total_classes)
            self.criterion = HierarchySemanticLoss(R=self.R, hierarchy=self.dependency_tree, dataset_name=self.settings.dataset)
        elif self.settings.model == 'none_all':
            model = NoSki(base_model=base_model, base_model_features=base_model_features, levels_labels=self.labels_levels, n_classes=self.total_classes)
            self.criterion = NoskiLoss(hierarchy=self.dependency_tree, to_eval=torch.tensor(self.dimensions_to_eval))
        elif self.settings.model == 'none_l0':
            model = NoSkiL0Only(base_model=base_model, base_model_features=base_model_features, n_classes=len(self.level_0_classes))
            self.criterion = NoSkiL0OnlyLoss()
        else:
            print('Specified model "{}" not recognized! '
                'Options are: resnet50 and wideresnet'.format(self.settings.model))
        # Move model to device
        self.visualize_model(model) if self.settings.visualize_model else None
        self.model = model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.logger.print_it('Model setup done!')

    def setup_training(self):
        self.logger.print_it('Setting up training with "{}" optimizer...'.format(self.settings.optimizer))

        # Setup optimizer
        params_lr_list = [{'params': params, 'lr': self.settings.lr, 'name': 'HEAD'} for params in self.model.get_head_parameters_list()] + \
                        [{'params': params, 'lr': self.settings.lr/10, 'name': 'BACKBONE'} for params in self.model.get_backbone_parameters_list()]
        if self.settings.optimizer == 'sgd':
            self.optimizer = SGD(params_lr_list,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        elif self.settings.optimizer == 'sam':
            self.optimizer = SAM(params_lr_list,
                                    base_optimizer=SGD,
                                    momentum=0.9,
                                    nesterov=False,
                                    rho=0.05)
        elif self.settings.optimizer == 'adam':
            self.optimizer = Adam(params_lr_list)
        else:
            raise ValueError('Specified optimizer "{}" not supported. Options are: adam and sgd and sam'.format(self.settings.optimizer))
        
        if self.settings.lr_sched == 'const':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1)
        elif self.settings.lr_sched == 'warmup_step':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(self.settings.epochs/3), gamma=0.1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(self.settings.epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif self.settings.lr_sched == 'warmup_exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=math.ceil(self.settings.epochs/40), after_scheduler=scheduler)
            self.scheduler.step()
        elif self.settings.lr_sched == 'warmup_cosine':
            cycle_steps = math.ceil(self.settings.epochs/5)
            warmup_steps = math.ceil(cycle_steps/10)
            max_lr=self.settings.lr
            min_lr=max_lr/100
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=cycle_steps, cycle_mult=1.0, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, gamma=0.5)
        elif self.settings.lr_sched == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=math.ceil(self.settings.epochs/3), gamma=0.1)
        elif self.settings.lr_sched == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        elif self.settings.lr_sched == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.settings.epochs)
        else:
            raise ValueError('Learning rate scheduler "{}" not available!'.format(self.settings.lr_sched))
        
        # Setting up metrics objects
        self.auprc = Auprc(hierarchy=self.dependency_tree)
        self.jaccard = Jaccard(hierarchy=self.dependency_tree)
        self.levels_acc = LevelsAcc(hierarchy=self.dependency_tree)
        # self.forgettability_met = Forgettability()
        # self.epochs_to_learn_met = EpochsToLearn()
        self.hessian_flatness_met = HessianTraceCurvature(model=self.model, loss_function=self.criterion, device=self.device)
        # self.epsilon_flatness_met = EpsilonFlatness(model=self.model, loss_function=self.criterion, device=self.device)

        self.logger.print_it('Training setup done!')

    def train(self):
        if self.settings.resume:
            self.load_last_resume_ckpt()
        else:
            # Initialize dictionary to save statistics for every example presentation
            self.setup_initial_metrics()
            self.logger.print_it('Starting training of "{}" for "{}" from scratch. This will take a while...'.format(self.settings.model,
                                                                                                                    self.settings.dataset))

        while(self.epoch <= self.settings.epochs):
            start_time = time.time()

            # Train and validation
            train_epoch_metrics = self.train_epoch()
            val_epoch_metrics = self.val_epoch()
            val_epoch_acc = val_epoch_metrics['lv_0_acc']
            self.train_metrics.append(train_epoch_metrics)
            self.val_metrics.append(val_epoch_metrics)
            self.val_accs.append(val_epoch_acc)

            epoch_time = time.time() - start_time
            self.elapsed_time += epoch_time
            h, m, s = Trainer.convert_to_hms(self.elapsed_time)
            self.logger.print_it('Elapsed time for epoch {}: {}:{:02d}:{:02d}'.format(self.epoch,h,m,s))

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint when best model
            if val_epoch_acc > self.best_acc:
                self.logger.print_it('New Best model at epoch {}: \t Top1-acc = {:.2f}'.format(self.epoch, val_epoch_acc*100))
                self.save_model('best.pt')
                self.best_acc = val_epoch_acc
            
            # Save model when last epoch
            if self.epoch == (self.settings.epochs):
                self.logger.print_it('Saving last model: \t Top1-acc = {:.2f}'.format(val_epoch_acc*100))
                self.save_model('last.pt')

            self.store_resume_ckpt()

            self.epoch += 1

        h, m, s = Trainer.convert_to_hms(self.elapsed_time)
        self.logger.print_it('Training of "{}" for "{}" completed in: {}:{:02d}:{:02d}'.format(self.settings.model, self.settings.dataset, h, m, s))
        # Compute metrics and store them in folder
        self.compute_and_store_final_metrics()

    def train_epoch(self):
        train_loss = 0.
        # Set model to train mode and reset metrics
        self.model.train()
        self.auprc.reset()
        self.jaccard.reset()
        self.levels_acc.reset()

        self.logger.print_it_log_file_only('Epoch {} Learning rates:'.format(self.epoch))
        for param_group in self.optimizer.param_groups:
            self.logger.print_it_log_file_only('Parameter group of "{}" -> LR = {}'.format(param_group['name'], param_group['lr']))

        # Iterate
        for batch_idx, batch in enumerate(self.train_loader):
            (inputs, labels) = batch
            # Map to available device
            inputs = inputs.to(self.device)
            labels = [label.to(self.device) for label in labels]
            # Compute loss and predictions
            if self.settings.optimizer=='sam':
                # first forward-backward step
                enable_running_stats(self.model)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.mean().backward()
                self.optimizer.first_step(zero_grad=True)
                # second forward-backward step
                disable_running_stats(self.model)
                self.criterion(self.model(inputs), labels).mean().backward()
                self.optimizer.second_step(zero_grad=True)  
            else:
                # Forward propagation, compute loss, get predictions
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            # Update loss, backward propagate, update optimizer
            # print('loss before mean in train: {}'.format(loss))
            # loss = loss.mean()
            # print('loss after mean in train: {}'.format(loss))
            train_loss += loss.item()
            if self.settings.optimizer != 'sam':
                loss.backward()
                if self.settings.model in ['chmcnn']:
                    self.model.float()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
            # Print message on console
            metrics = {'loss': train_loss / (batch_idx+1),
                        'auprc': self.auprc.compute_quick(self.model.auprc_predictions_from_outputs(outputs), labels, logger=self.logger),
                        'jaccard' : self.jaccard.compute_quick(self.model.one_hot_predictions_from_outputs(outputs), labels),}
            metrics.update(self.levels_acc.compute_quick(self.model.levelled_predictions_from_outputs(outputs), labels))
            self.print_message(epoch_index=self.epoch, total_epochs=self.settings.epochs,
                            index_batch=batch_idx+1, total_batches=len(self.train_loader),
                            metrics=metrics, mode='train')
        self.logger.set_logger_newline()
        return metrics

    def val_epoch(self):
        with torch.no_grad():
            val_loss = 0.
            # Set model to evaluation mode and reset metrics
            self.model.eval()
            self.auprc.reset()
            self.jaccard.reset()
            self.levels_acc.reset()
            # Iterate
            for batch_idx, batch in enumerate(self.val_loader):
                (inputs, labels) = batch
                # Map to available device
                inputs = inputs.to(self.device)
                labels = [label.to(self.device) for label in labels]
                # Compute output and loss
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = loss.mean()
                val_loss += loss.item()
                # Print message on console
                metrics = {'loss': val_loss / (batch_idx + 1),
                            'auprc': self.auprc.compute(self.model.auprc_predictions_from_outputs(outputs), labels),
                            'jaccard' : self.jaccard.compute(self.model.one_hot_predictions_from_outputs(outputs), labels),}
                metrics.update(self.levels_acc.compute(self.model.levelled_predictions_from_outputs(outputs), labels))
                self.print_message(epoch_index=self.epoch, total_epochs=self.settings.epochs,
                                index_batch=batch_idx+1, total_batches=len(self.val_loader),
                                metrics=metrics, mode='val')
        self.logger.set_logger_newline()
        return metrics

    def print_message(self, epoch_index, total_epochs, index_batch, total_batches, metrics, mode='train'):
        message = '| EPOCH: {}/{} |'.format(epoch_index, total_epochs)
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
                train_metrics_message += '{}={:.5f}{} '.format(metric_name, metric_value, ',' if index < len(metrics.keys()) - 1 else '')
                index += 1
            message += train_metrics_message
        message += '|'
        self.logger.print_it_same_line(message)

    def compute_and_store_final_metrics(self):
        metrics_dir = os.path.join(self.settings.metrics_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)
        # # Forgettability metrics
        # forgettability_train = self.forgettability_met.get_scores(mode='train')
        # with open(os.path.join(metrics_dir, 'forgettability_train.pkl'), 'wb') as file:
        #     pickle.dump(forgettability_train, file)
        # del forgettability_train
        # forgettability_test = self.forgettability_met.get_scores(mode='test')
        # with open(os.path.join(metrics_dir, 'forgettability_test.pkl'), 'wb') as file:
        #     pickle.dump(forgettability_test, file)
        # del forgettability_test
        # self.forgettability_met.store_history(os.path.join(metrics_dir, 'forgettability_history.pkl'))
        # del self.forgettability_met
        # # Epochs to learn metrics
        # epochs_to_learn_train = self.epochs_to_learn_met.get_scores(mode='train')
        # with open(os.path.join(metrics_dir, 'epochs_to_learn_train.pkl'), 'wb') as file:
        #     pickle.dump(epochs_to_learn_train, file)
        # del epochs_to_learn_train
        # epochs_to_learn_test = self.epochs_to_learn_met.get_scores(mode='test')
        # with open(os.path.join(metrics_dir, 'epochs_to_learn_test.pkl'), 'wb') as file:
        #     pickle.dump(epochs_to_learn_test, file)
        # del epochs_to_learn_test
        # self.epochs_to_learn_met.store_history(os.path.join(metrics_dir, 'epochs_to_learn_history.pkl'))
        # del self.epochs_to_learn_met
        # Hessian flatness metrics
        hessian_flat_train = self.hessian_flatness_met.compute_dataset(self.train_loader)
        with open(os.path.join(metrics_dir, 'hessian_flat_train.pkl'), 'wb') as file:
            pickle.dump(hessian_flat_train, file)
        del hessian_flat_train
        hessian_flat_test = self.hessian_flatness_met.compute_dataset(self.val_loader)
        with open(os.path.join(metrics_dir, 'hessian_flat_test.pkl'), 'wb') as file:
            pickle.dump(hessian_flat_test, file)
        del hessian_flat_test
        # Epsilon flatness metrics
        # epsilon_flat_train = self.epsilon_flatness_met.compute_dataset(self.train_dataset, sample_indices=self.train_indx)
        # with open(os.path.join(metrics_dir, 'epsilon_flat_train.pkl'), 'wb') as file:
        #     pickle.dump(epsilon_flat_train, file)
        # del epsilon_flat_train
        # epsilon_flat_test = self.epsilon_flatness_met.compute_dataset(self.val_dataset, sample_indices=self.val_indx)
        # with open(os.path.join(metrics_dir, 'epsilon_flat_test.pkl'), 'wb') as file:
        #     pickle.dump(epsilon_flat_test, file)
        # del epsilon_flat_test

    
    def store_resume_ckpt(self):
        state = {
                'epoch': self.epoch,
                'arch': self.settings.model,
                'best_acc': self.best_acc,
                'elapsed_time': self.elapsed_time,
                'val_accs': self.val_accs,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'scheduler' : self.scheduler.state_dict(),
            }
        checkpoint_folder = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        os.makedirs(checkpoint_folder, exist_ok=True)
        torch.save(state, os.path.join(checkpoint_folder, 'epoch={}.pth.tar'.format(self.epoch)))


    def load_last_resume_ckpt(self):
        checkpoint_folder = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        checkpoints = glob.glob(os.path.join(checkpoint_folder, '*.pth.tar'))
        found_epochs = [int(check.split('epoch=')[-1].split('.pth.tar')[0]) for check in checkpoints]
        if found_epochs == []:
            self.logger.print_it('No resume checkpoint found! Are you sure you wanted to resume?')
            self.setup_initial_metrics()
            self.logger.print_it('Starting training of "{}" for "{}" from scratch. This will take a while...'.format(self.settings.model,
                                                                                                                    self.settings.dataset))
        else:
            last_epoch = max(found_epochs)
            checkpoint_to_load = os.path.join(self.settings.resume_ckpts_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed), 'epoch={}.pth.tar'.format(last_epoch))
            # Read variables
            checkpoint = torch.load(checkpoint_to_load, weights_only=False, map_location=self.device)
            self.epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
            self.elapsed_time = checkpoint['elapsed_time']
            self.val_accs = checkpoint['val_accs']
            self.train_metrics = checkpoint['train_metrics']
            self.val_metrics = checkpoint['val_metrics']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def setup_initial_metrics(self):
        self.best_acc = 0
        self.elapsed_time = 0
        self.val_accs = []
        self.train_metrics = []
        self.val_metrics = []
        self.epoch = 1

    def visualize_model(self, model):
        visualization_directory = os.path.join(self.settings.models_visualization_folder, self.settings.model, self.settings.dataset)
        visualization_filename = 'model'
        os.makedirs(visualization_directory, exist_ok=True)
        model_graph = draw_graph(model, input_size=(1, 3, self.img_size[0], self.img_size[1]), expand_nested=True, save_graph=True, filename=visualization_filename, directory=visualization_directory)
        model_graph.resize_graph(scale=5.0)
        model_graph.visual_graph.render(format='png', filename=visualization_filename, directory=visualization_directory)
        self.logger.print_it('Model visualization completed successfully! File stored at {}.png'.format(os.path.join(visualization_directory, visualization_filename)))

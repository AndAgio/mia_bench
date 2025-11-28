import os
import pickle
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from src.datasets import AircraftDataset, CIFAR100, CarsDataset, CubDataset, Fashion
from src.models.no_ski import NoskiLoss
from src.models.no_ski_l0_only import NoSkiL0OnlyLoss
from src.models.hrn import TreeLoss
from src.models.multiplexnet import MultiPlexLoss, logic_terms
from src.models.chmcnn import ChmcnnLoss
from src.models.spl import SplLoss
from src.models.semloss import HierarchySemanticLoss
from src.metrics.performance import Auprc, Jaccard, LevelsAcc
from src.metrics.flatness import HessianTraceCurvature, EpsilonFlatness
from .runner import Runner


class Tester(Runner):
    def __init__(self, settings):
        super().__init__(settings=settings, experiment_name='test_{}_with_{}_over_{}'.format(settings.model, settings.base_model, settings.dataset))
        self.gather_dataset()
        self.setup_model()
        self.setup_loss()
        self.setup_metrics()    

    def setup_model(self):
        try:
            self.load_best_model()
        except FileNotFoundError:
            try:
                self.load_last_model()
            except FileNotFoundError:
                raise FileNotFoundError('Best or last model should be available to run test!')
            
    def setup_loss(self):
        self.logger.print_it('Setting up loss for model "{}"...'.format(self.settings.model))
        if self.settings.model == 'hrn':
            self.criterion = TreeLoss(hierarchy=self.dependency_tree).to(self.device)
        elif self.settings.model == 'multiplexnet':
            terms = logic_terms(level_0_classes=self.level_0_classes,
                                level_1_classes=self.level_1_classes,
                                level_0_level_1_mapping=self.level_0_level_1_mapping)
            self.criterion = MultiPlexLoss(terms=terms, classes=self.level_0_classes).to(self.device)
        elif self.settings.model == 'chmcnn':
            self.criterion = ChmcnnLoss(R=self.R, hierarchy=self.dependency_tree)
        elif self.settings.model == 'spl':
            self.criterion = SplLoss(cmpe=self.model.cmpe, hierarchy=self.dependency_tree)
        elif self.settings.model == 'sem_loss':
            self.criterion = HierarchySemanticLoss(R=self.R, hierarchy=self.dependency_tree, dataset_name=self.settings.dataset)
        elif self.settings.model == 'none_all':
            self.criterion = NoskiLoss(hierarchy=self.dependency_tree)
        elif self.settings.model == 'none_l0':
            self.criterion = NoSkiL0OnlyLoss()
        else:
            print('Specified model "{}" not recognized!'.format(self.settings.model))
        self.criterion = self.criterion.to(self.device)
        self.logger.print_it('Model setup done!')

    def gather_dataset(self):
        self.logger.print_it('Gathering dataset "{}". This may take a while...'.format(self.settings.dataset))
        # Load the appropriate train and test datasets
        chosen_dataset = self.settings.dataset
        # Data transforms (normalization & data augmentation)
        if chosen_dataset in ['std_cars', 'aircrafts', 'cub_birds', 'fashion']:
            stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if chosen_dataset == 'fahsion' else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            self.img_size = (448, 448)
            test_trans = T.Compose([T.Resize(self.img_size), T.ToTensor(), T.Normalize(*stats)])
        elif chosen_dataset == 'cifar':
            stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            self.img_size = (256, 256)
            test_trans = T.Compose([T.Resize(self.img_size), T.ToTensor(), T.Normalize(*stats)])
        else:
            raise ValueError('Dataset "{}" is not available!'.format(chosen_dataset))
        # Load dataset
        if chosen_dataset == 'std_cars':
            test_data = CarsDataset("data/std_cars", train=False, transform=test_trans, download=True)
        elif chosen_dataset == 'aircrafts':
            test_data = AircraftDataset("data/aircrafts", train=False, transform=test_trans, download=True)
        elif chosen_dataset == 'cub_birds':
            test_data = CubDataset("data/cub_birds", train=False, transform=test_trans, download=True)
        elif chosen_dataset == 'cifar':
            test_data = CIFAR100('data/cifar100', train=False, transform=test_trans, download=True)
        elif chosen_dataset == 'fashion':
            test_data = Fashion("data/fashion", train=False, transform=test_trans, download=True)
        else:
            raise ValueError('Dataset "{}" is not available!'.format(chosen_dataset))
        self.dependency_tree = test_data.get_class_dependency_tree()
        self.adj_matrix = test_data.get_adjecency_matrix()
        self.R = test_data.get_R()
        self.n_levels = test_data.get_n_levels()
        self.labels_levels = test_data.get_labels_levels()
        self.level_0_classes = test_data.get_level_0_classes()
        self.level_1_classes = test_data.get_level_1_classes()
        self.level_0_level_1_mapping = test_data.get_level_0_level_1_mapping()
        self.total_classes = test_data.get_n_total_classes()
        # Generate loaders
        self.test_loader = DataLoader(test_data, batch_size=self.settings.batch_size)
        self.logger.print_it('Gathered dataset "{}"'.format(self.settings.dataset))

    def setup_metrics(self):
        # Setting up metrics objects
        self.auprc = Auprc(hierarchy=self.dependency_tree)
        self.jaccard = Jaccard(hierarchy=self.dependency_tree)
        self.levels_acc = LevelsAcc(hierarchy=self.dependency_tree)
        self.hessian_flatness_met = HessianTraceCurvature(model=self.model, loss_function=self.criterion, device=self.device)
        self.epsilon_flatness_met = EpsilonFlatness(model=self.model, loss_function=self.criterion, device=self.device)

    def test(self, mode='best', compute_flat=False):
        if mode == 'best':
            self.load_best_model()
        elif mode == 'last':
            self.load_last_model()
        else:
            raise ValueError('Mode "{}" for running Tester.test() is not available!'.format(mode))
        self.logger.print_it('Testing of "{}" for "{}"...'.format(self.settings.model, self.settings.dataset))
        with torch.no_grad():
            # Set model to evaluation mode and reset metrics
            self.model.eval()
            self.auprc.reset()
            self.jaccard.reset()
            self.levels_acc.reset()
            # Iterate
            for batch_idx, batch in enumerate(self.test_loader):
                (inputs, labels) = batch
                # Map to available device
                inputs = inputs.to(self.device)
                labels = [label.to(self.device) for label in labels]
                # Compute output and loss
                outputs = self.model(inputs)
                # Print message on console
                metrics = {'auprc': self.auprc.compute(self.model.auprc_predictions_from_outputs(outputs), labels),
                            'jaccard': self.jaccard.compute(self.model.one_hot_predictions_from_outputs(outputs), labels),}
                metrics.update(self.levels_acc.compute(self.model.levelled_predictions_from_outputs(outputs), labels))
                self.print_message(index_batch=batch_idx+1, total_batches=len(self.test_loader),
                                metrics=metrics, mode='test')
        self.logger.set_logger_newline()
        if compute_flat:
            self.compute_flatness()
        self.store_results(metrics)
        return metrics['lv_0_acc']

    def print_message(self, index_batch, total_batches, metrics, mode='test'):
        message = '|'
        bar_length = 20
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

    def compute_flatness(self):
        metrics_dir = os.path.join(self.settings.metrics_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)
        # Hessian flatness metrics
        hessian_flat_test = self.hessian_flatness_met.compute_dataset(self.test_loader)
        with open(os.path.join(metrics_dir, 'hessian_flat_test.pkl'), 'wb') as file:
            pickle.dump(hessian_flat_test, file)
        del hessian_flat_test
        # Epsilon flatness metrics
        epsilon_flat_test = self.epsilon_flatness_met.compute_dataset(self.test_loader)
        with open(os.path.join(metrics_dir, 'epsilon_flat_test.pkl'), 'wb') as file:
            pickle.dump(epsilon_flat_test, file)
        del epsilon_flat_test

    def store_results(self, metrics):
        metrics_dir = os.path.join(self.settings.metrics_folder, self.experiment_name, 'seed_{}'.format(self.settings.seed))
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, 'performances.txt'), 'x') as file:
            for perf_key, perf_value in metrics.items():
                file.write('{} = {}\n'.format(perf_key, perf_value))
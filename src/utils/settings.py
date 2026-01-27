import argparse
import json
import os
import pathlib
from src.utils.configs import generate_configs_from_settings, get_hash_from_settings, get_relevant_settings
from .variables import DEFAULT_DATASETS_FOLDER, DEFAULT_LOG_FOLDER, DEFAULT_MODELS_FOLDER, DEFAULT_RESUME_CKPTS_FOLDER, DEFAULT_METRICS_FOLDER, DEFAULT_OUT_FOLDER, DEFAULT_PLOTS_FOLDER


def gather_settings():
        # Training settings
        parser = argparse.ArgumentParser(description='MIA Benchmarking code')
        
        # Dataset parameters
        parser.add_argument("--dataset", type=str, default="cifar100",
                                choices=["cifar10", "cifar100", "svhn", "fmnist", "cinic10", "imagenet", "tinyimagenet"])
        
        # Model parameters
        parser.add_argument("--defender_model", default="resnet18",
                                choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                                        'wideresnet_16_8', 'wideresnet_28_2', 'wideresnet_28_10', 'wideresnet_50_2', 'wideresnet_101_2',
                                        "inception_v3", 
                                        "vgg11", "vgg13", "vgg16", "vgg19", 
                                        "mobile_small", "mobile_large", 
                                        "vit"])
        # parser.add_argument("--visualize_model", action="store_true", default=False,
        #                     help="whether to visualize the plot of the NN or not (for debugging)",)

        # Training parameters 
        parser.add_argument('--defender_optimizer', type=str, required=False, default='sgd',
                                help='optimizer')
        parser.add_argument('--defender_epochs', type=int, default=100,
                                help='Max number of epochs to train')
        parser.add_argument('--defender_batch_size', type=int, required=False, default=256,
                                help='input batch size for training')
        parser.add_argument('--defender_loss', type=str, required=False, default='crossentropy',
                                help='loss to be used for training', choices=['crossentropy'])
        parser.add_argument('--perf_metrics', type=str, nargs="+", required=False, default=['accuracy'],
                                help='performance metrics to be used for training', choices=['accuracy', 'mse', 'mae', 'rmse'])
        parser.add_argument('--perf_metric_to_track', type=str, required=False, default='accuracy',
                                help='performance metric to track best model while training', choices=['accuracy', 'mse', 'mae', 'rmse'])
        parser.add_argument('--defender_lr', type=float, required=False, default=0.01,
                                help='learning rate')
        parser.add_argument('--defender_lr_sched', type=str, required=False, default='cosine',
                                help='lr scheduler', choices=['const', 'step', 'multistep', 'exp', 'cosine', 'warmup_step', 'warmup_exp', 'warmup_cosine'])
        parser.add_argument('--defender_lr_step_size', type=int, required=False, default=50,
                                help='step size in epochs for the step, warmup step lr schedulers')
        parser.add_argument('--defender_lr_step_gamma', type=float, required=False, default=0.1,
                                help='decrease multiplying factor for the step, warmup step lr schedulers')
        parser.add_argument('--defender_lr_warmup_multiplier', type=float, required=False, default=1,
                                help='multiplicative factor for the warmup phase of warmup step, warmup exp, warmup cosine lr schedulers')
        parser.add_argument('--defender_lr_warmup_epochs', type=int, required=False, default=10,
                                help='number of epochs to use as warmup in warmup step, warmup exp, warmup cosine lr schedulers')
        parser.add_argument('--defender_lr_exp_gamma', type=float, required=False, default=0.98,
                                help='decrease multiplying factor for the exp, warmup exp lr schedulers')
        parser.add_argument('--defender_lr_cycle_step', type=int, required=False, default=40,
                                help='number of epochs in each warmup and restart cycle of the warmup cosine lr schedulers')
        parser.add_argument('--defender_lr_cycle_gamma', type=float, required=False, default=1,
                                help='dacaying factor to be applied in each warmup and restart cycle of the warmup cosine lr schedulers')
        parser.add_argument('--defender_lr_cosine_min', type=float, required=False, default=0.0001,
                                help='minmum learning rate to use in warmup cosine and cosine lr schedulers')
        parser.add_argument('--defender_lr_step_milestones', nargs="+", type=int, default=[60, 120],
                                help='Set of milestones to be used to decay lr in multistep lr scheduler')

        parser.add_argument('--defender_weight_decay', type=float, required=False, default=5e-4,
                                help='weight decay')
        parser.add_argument('--defender_momentum', type=float, required=False, default=0.9,
                                help='momentum')
        parser.add_argument('--defender_nesterov', action="store_true", default=False,
                                help='nesterov')
        parser.add_argument('--defender_seed', type=int, default=12345,
                                help='random seed (default:12345)')
        
        # Shared MIA parameters
        parser.add_argument("--defense_mode", default="none",
                                choices=['none', 'no', 'vanilla',
                                        "dp", "differential_privacy", "differential-privacy", 
                                        'mem_guard', 'memguard', 'mem-guard',
                                        'data_augmentation', 'augmentation', 'data-augmentation',])
        # Differential Privacy parameters for defender model
        # parser.add_argument("--defender_use_dp", action="store_true", default=False,
        #                         help="enable Differential Privacy for defender model training",)
        parser.add_argument('--defender_dp_noise_multiplier', type=float, default=1.0,
                                help='Noise multiplier for DP-SGD')
        parser.add_argument('--defender_dp_max_grad_norm', type=float, default=1.0,
                                help='Max grad norm for DP-SGD')
        parser.add_argument("--defender_dp_clip_per_layer", action="store_true", default=False,
                                help="whether to use per layer clipping in DP-SGD",)
        parser.add_argument("--defender_dp_grad_sample_mode", type=str, default="ghost")
        # MemGuard parameters for defender model
        parser.add_argument('--defender_mem_guard_budget', type=float, default=0.1,
                                help='Budget parameter for MemGuard defense')
        parser.add_argument('--defender_mem_guard_shadow_model_layers', type=int, nargs="+", default=[64, 32],
                                help='List of hidden layer sizes for the shadow attacker model in MemGuard defense')
        parser.add_argument('--defender_mem_guard_shadow_model_epochs', type=int, default=30,
                                help='Number of epochs to train the shadow attacker model in MemGuard defense')
        parser.add_argument('--defender_mem_guard_shadow_model_lr', type=float, default=0.01,
                                help='Learning rate to train the shadow attacker model in MemGuard defense')
        # Data Augmentation parameters for defender model
        parser.add_argument("--data_augmentation", action="store_true", default=False,
                                help="augment data by flipping and cropping",)
        
        
        # Hardware related settings
        parser.add_argument("--device", default='0',
                                help="Set to 0 or 1 to enable CUDA training, cpu otherwise")
        parser.add_argument("--distributed", action="store_true", default=False,
                                help="use distributed training options",)
        
        # Folders parameters
        parser.add_argument('--datasets_folder', type=pathlib.Path, default=DEFAULT_DATASETS_FOLDER)
        parser.add_argument('--log_folder', type=pathlib.Path, default=DEFAULT_LOG_FOLDER)
        parser.add_argument('--models_folder', type=pathlib.Path, default=DEFAULT_MODELS_FOLDER)
        parser.add_argument('--resume_ckpts_folder', type=pathlib.Path, default=DEFAULT_RESUME_CKPTS_FOLDER)
        parser.add_argument('--metrics_folder', type=pathlib.Path, default=DEFAULT_METRICS_FOLDER)
        parser.add_argument('--out_folder', type=pathlib.Path, default=DEFAULT_OUT_FOLDER)
        parser.add_argument('--plots_folder', type=pathlib.Path, default=DEFAULT_PLOTS_FOLDER)
        # parser.add_argument('--models_visualization_folder', type=pathlib.Path, default='models_graphviz')

        # parser.add_argument('--resume_ckpts_folder', type=str, default='resume_ckpts')
        parser.add_argument("--resume", action="store_true", default=False,
                                help="resume training from last checkpoint found",)
        
        

        # Shared MIA parameters
        parser.add_argument("--attack_mode", default="quantile",
                                choices=['online_robust', 'offline_robust', 'on_robust', 'off_robust', "lira", "quantile",
                                        "neural_feat", "neural_prob", "neural_logit", 
                                        'rmia_loss', 'rmia_confidence', 'rmia_entropy', 
                                        'pmia_loss', 'pmia_confidence', 'pmia_entropy'])
        parser.add_argument('--n_auditing_samples', type=int, default=1000,
                                help='Number of samples to use for auditing on the attacker side')
        parser.add_argument('--audit_in_perc', type=float, default=0.5,
                                help='Percentage of auditing samples that are coming from the training set')
        parser.add_argument('--n_shadows', type=int, default=10,
                                help='Number of shadow models and datasets to be used for MIAs requiring shadow models')
        parser.add_argument('--n_samples_per_shadow_dataset', type=int, default=5000,
                                help='Number of samples to use for each shadow datasets on the attacker side')
        parser.add_argument('--shadow_test_perc', type=float, default=0.5,
                                help='Percentage of shadow dataset samples that are coming from the testing set')
        # RobustMIA parameters
        parser.add_argument('--random_population_size', type=int, default=1000,
                                help='Number of samples in Z to select randomly for LR computation')
        parser.add_argument('--robust_alphas', nargs="+", type=float, default=0.5,
                                help='Set of alphas to be used in the RobustMIA attack')
        parser.add_argument('--robust_gamma', type=float, default=1,
                                help='Gamma value to be used in the RobustMIA attack')
        # Quantile MIA parameters
        parser.add_argument('--n_quantile', type=int, default=100,
                                help='Number of quantiles')
        parser.add_argument('--low_quantile', type=float, default=0.01,
                                help='Lowest quantile in the quantile MIA attack')
        parser.add_argument('--high_quantile', type=float, default=0.99,
                                help='Highest quantile in the quantile MIA attack')
        parser.add_argument('--quantile_alpha', type=float, default=0.05,
                                help='Alpha to be used in the Quantile MIA attack')
        parser.add_argument("--quantile_use_logscale", action="store_true", default=False,
                                help="use logscale for quantile MIA attack",)
        parser.add_argument("--quantile_use_gaussian", action="store_true", default=False,
                                help="use use_gaussian for quantile MIA attack",)
        # Neural MIA parameters
        parser.add_argument('--neural_model_layers', type=int, nargs="+", default=[64, 32],
                                help='List of hidden layer sizes for the neural MIA attacker model')
        parser.add_argument('--neural_model_epochs', type=int, default=20,
                                help='Number of epochs to train the neural MIA attacker model')
        parser.add_argument('--neural_model_lr', type=float, default=0.01,
                                help='Learning rate to train the neural MIA attacker model')
        # Attack-R MIA parameters
        parser.add_argument('--r_alpha', type=float, default=0.05,
                                help='Alpha to be used in the Attack-R MIA attack')
        # Attack-P MIA parameters
        parser.add_argument('--p_alpha', type=float, default=0.05,
                                help='Alpha to be used in the Attack-P MIA attack')
        

        # Attacker training parameters
        parser.add_argument("--att_model", default="resnet18",
                                choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                                        'wideresnet_16_8', 'wideresnet_28_2', 'wideresnet_28_10', 'wideresnet_50_2', 'wideresnet_101_2',
                                        "inception_v3", 
                                        "vgg11", "vgg13", "vgg16", "vgg19", 
                                        "mobile_small", "mobile_large", 
                                        "vit"])
        parser.add_argument('--att_optimizer', type=str, required=False, default='sgd',
                                help='optimizer')
        parser.add_argument('--att_epochs', type=int, default=100,
                                help='Max number of epochs to train')
        parser.add_argument('--att_batch_size', type=int, required=False, default=256,
                                help='input batch size for training')
        parser.add_argument('--att_loss', type=str, required=False, default='crossentropy',
                                help='loss to be used for training', choices=['crossentropy'])
        parser.add_argument('--att_lr', type=float, required=False, default=0.01,
                                help='learning rate')
        parser.add_argument('--att_lr_sched', type=str, required=False, default='cosine',
                                help='lr scheduler', choices=['const', 'step', 'exp', 'cosine', 'warmup_step', 'warmup_exp', 'warmup_cosine'])
        parser.add_argument('--att_lr_step_size', type=int, required=False, default=50,
                                help='step size in epochs for the step, warmup step lr schedulers')
        parser.add_argument('--att_lr_step_gamma', type=float, required=False, default=0.1,
                                help='decrease multiplying factor for the step, warmup step lr schedulers')
        parser.add_argument('--att_lr_warmup_multiplier', type=float, required=False, default=1,
                                help='multiplicative factor for the warmup phase of warmup step, warmup exp, warmup cosine lr schedulers')
        parser.add_argument('--att_lr_warmup_epochs', type=int, required=False, default=10,
                                help='number of epochs to use as warmup in warmup step, warmup exp, warmup cosine lr schedulers')
        parser.add_argument('--att_lr_exp_gamma', type=float, required=False, default=0.98,
                                help='decrease multiplying factor for the exp, warmup exp lr schedulers')
        parser.add_argument('--att_lr_cycle_step', type=int, required=False, default=40,
                                help='number of epochs in each warmup and restart cycle of the warmup cosine lr schedulers')
        parser.add_argument('--att_lr_cycle_gamma', type=float, required=False, default=1,
                                help='dacaying factor to be applied in each warmup and restart cycle of the warmup cosine lr schedulers')
        parser.add_argument('--att_lr_cosine_min', type=float, required=False, default=0.0001,
                                help='minmum learning rate to use in warmup cosine and cosine lr schedulers')
        parser.add_argument('--att_lr_step_milestones', nargs="+", type=int, default=[60, 120],
                                help='Set of milestones to be used to decay lr in multistep lr scheduler')

        parser.add_argument('--att_weight_decay', type=float, required=False, default=5e-4,
                                help='weight decay')
        parser.add_argument('--att_momentum', type=float, required=False, default=0.9,
                                help='momentum')
        parser.add_argument('--att_nesterov', action="store_true", default=False,
                                help='nesterov')
        parser.add_argument('--att_seed', type=int, default=12345,
                                help='random seed (default:12345)')
        # Differential Privacy parameters for attacker model
        # parser.add_argument("--att_use_dp", action="store_true", default=False,
        #                         help="enable Differential Privacy for attacker model training",)
        # parser.add_argument('--att_dp_noise_multiplier', type=float, default=1.0,
        #                         help='Noise multiplier for DP-SGD')
        # parser.add_argument('--att_dp_max_grad_norm', type=float, default=1.0,
        #                         help='Max grad norm for DP-SGD')
        # parser.add_argument("--att_dp_clip_per_layer", action="store_true", default=False,
        #                         help="whether to use per layer clipping in DP-SGD",)
        # parser.add_argument("--att_dp_grad_sample_mode", type=str, default="ghost")
        

        settings = parser.parse_args()
        return settings


def setup_configs_and_folder_from_settings(settings):
        for mode in ['defender', 'attacker', 'experiment']:
                hash_code = get_hash_from_settings(settings, mode=mode)
                print(f"{mode.capitalize()} hash: {hash_code}")
                out_folder = settings.out_folder / f"{mode}s" / hash_code
                if mode == 'experiment':
                        experiment_out_folder = out_folder
                os.makedirs(out_folder, exist_ok=True)
                settings_file = out_folder/'settings.json'
                with open(settings_file, 'w') as f:
                        settings_dict = get_relevant_settings(settings, mode=mode)
                        json.dump(settings_dict, f, indent=4)
                print(f"{mode.capitalize()} settings saved to {settings_file}")
        experiment_configs = generate_configs_from_settings(settings)
        return experiment_configs, experiment_out_folder
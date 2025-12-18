import argparse
import pathlib
from .variables import DEFAULT_DATASETS_FOLDER, DEFAULT_LOG_FOLDER, DEFAULT_MODELS_FOLDER, DEFAULT_RESUME_CKPTS_FOLDER, DEFAULT_METRICS_FOLDER, DEFAULT_OUT_FOLDER, DEFAULT_PLOTS_FOLDER
    

def gather_settings():
    # Training settings
    parser = argparse.ArgumentParser(description='MIA Benchmarking code')
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar10", "cifar100", "svhn", "fmnist", "imagenet", "tiny_imagenet"])
    
    # Model parameters
    parser.add_argument("--victim_model", default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                                'wideresnet_16_8', 'wideresnet_28_2', 'wideresnet_28_10', 'wideresnet_50_2', 'wideresnet_101_2',
                                "inception_v3", 
                                "vgg11", "vgg13", "vgg16", "vgg19", 
                                "mobile_small", "mobile_large", 
                                "vit"])
    # parser.add_argument("--visualize_model", action="store_true", default=False,
    #                     help="whether to visualize the plot of the NN or not (for debugging)",)

    # Training parameters 
    parser.add_argument('--victim_optimizer', type=str, required=False, default='sgd',
                        help='optimizer')
    parser.add_argument('--victim_epochs', type=int, default=100,
                        help='Max number of epochs to train')
    parser.add_argument('--victim_batch_size', type=int, required=False, default=256,
                        help='input batch size for training')
    parser.add_argument('--victim_loss', type=str, required=False, default='crossentropy',
                        help='loss to be used for training', choices=['crossentropy'])
    parser.add_argument('--perf_metrics', type=str, nargs="+", required=False, default=['accuracy'],
                        help='performance metrics to be used for training', choices=['accuracy', 'mse', 'mae', 'rmse'])
    parser.add_argument('--perf_metric_to_track', type=str, required=False, default='accuracy',
                        help='performance metric to track best model while training', choices=['accuracy', 'mse', 'mae', 'rmse'])
    parser.add_argument('--victim_lr', type=float, required=False, default=0.001,
                        help='learning rate')
    parser.add_argument('--victim_lr_sched', type=str, required=False, default='cosine',
                        help='lr scheduler', choices=['const', 'step', 'exp', 'cosine', 'warmup_step', 'warmup_exp', 'warmup_cosine'])
    parser.add_argument('--victim_weight_decay', type=float, required=False, default=1e-5,
                        help='weight decay')
    parser.add_argument('--victim_seed', type=int, default=12345,
                        help='random seed (default:12345)')
    parser.add_argument("--use_grad_scaling", action="store_true", default=False,
                        help="use gradient scaling in training",)
    
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
    parser.add_argument("--resume", action="store_true", default=True,
                        help="resume training from last checkpoint found",)
    
    parser.add_argument("--data_augmentation", action="store_true", default=True,
                        help="augment data by flipping and cropping",)
    

    # Shared MIA parameters
    parser.add_argument("--attack_mode", default="rmia",
                        choices=['online_rmia', 'offline_rmia', 'on_rmia', 'off_rmia', "lira", "nn", "quantile"])
    parser.add_argument('--n_auditing_samples', type=int, default=1000,
                        help='Number of samples to use for auditing on the attacker side')
    parser.add_argument('--audit_in_perc', type=float, default=0.5,
                        help='Percentage of auditing samples that are coming from tßhe training set')
    parser.add_argument('--n_shadows', type=int, default=10,
                        help='Number of shadow models and datasets to be used for MIAs requiring shadow models')
    parser.add_argument('--n_samples_per_shadow_dataset', type=int, default=5000,
                        help='Number of samples to use for each shadow datasets on the attacker side')
    parser.add_argument('--shadow_test_perc', type=float, default=0.5,
                        help='Percentage of shadow dataset samples that are coming from the testing set')
    # RMIA parameters
    parser.add_argument('--random_population_size', type=int, default=1000,
                        help='Number of samples in Z to select randomly for LR computation')
    parser.add_argument('--rmia_alphas', nargs="+", type=float, default=0.5,
                        help='Set of alphas to be used in the RMIA attack')
    parser.add_argument('--rmia_gamma', type=float, default=1,
                        help='Gamma value to be used in the RMIA attack')
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
    parser.add_argument('--att_lr', type=float, required=False, default=0.001,
                        help='learning rate')
    parser.add_argument('--att_lr_sched', type=str, required=False, default='cosine',
                        help='lr scheduler', choices=['const', 'step', 'exp', 'cosine', 'warmup_step', 'warmup_exp', 'warmup_cosine'])
    parser.add_argument('--att_weight_decay', type=float, required=False, default=1e-5,
                        help='weight decay')
    parser.add_argument('--att_seed', type=int, default=12345,
                        help='random seed (default:12345)')
    

    
    settings = parser.parse_args()
    return settings

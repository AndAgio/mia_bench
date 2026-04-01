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
                                choices=["cifar10", "cifar100", "svhn", "fmnist", "cinic10", "imagenet", "tinyimagenet",
                                        "purchase", "texas", "news"],)
        parser.add_argument("--val_split", type=float, default=0.2,
                                help="percentage of training data to use for validation during defender model training",)
        
        # Model parameters
        parser.add_argument("--defender_model", default="resnet18",
                                choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                                        'wideresnet_16_8', 'wideresnet_28_2', 'wideresnet_28_10', 'wideresnet_50_2', 'wideresnet_101_2',
                                        "inception_v3", 
                                        "vgg11", "vgg13", "vgg16", "vgg19", 
                                        "mobile_small", "mobile_large", 
                                        "vit",
                                        "tabular_mlp",],)
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
                                        'relax_loss', 'relaxloss', 'relax-loss',
                                        'adv_reg', 'advreg', 'adv-reg',
                                        'mixup',
                                        'hamp', 'hamp_train', 'hamp_test', 'hamp_full',
                                        'selena',
                                        'mist', 'mist_mixup', 'mist-mixup',
                                        'weighted_smoothing', 'weighted-smoothing', 'weighted_smooth', 'weighted-smooth', 'weightedsmoothing', 'weightedsmooth', 'ws',
                                        'purifier',
                                        'mmd', 'mmd_mixup', 'mmd-mixup',
                                        'ldl',
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
        parser.add_argument("--defender_dp_grad_sample_mode", type=str, default="ghost", choices=["ghost", "hooks"],)
        # MemGuard parameters for defender model
        parser.add_argument('--defender_mem_guard_budget', type=float, default=0.1,
                                help='Budget parameter for MemGuard defense')
        parser.add_argument('--defender_mem_guard_shadow_model_layers', type=int, nargs="+", default=[64, 32],
                                help='List of hidden layer sizes for the shadow attacker model in MemGuard defense')
        parser.add_argument('--defender_mem_guard_shadow_model_epochs', type=int, default=30,
                                help='Number of epochs to train the shadow attacker model in MemGuard defense')
        parser.add_argument('--defender_mem_guard_shadow_model_lr', type=float, default=0.01,
                                help='Learning rate to train the shadow attacker model in MemGuard defense')
        # Relax Loss parameters for defender model
        parser.add_argument('--defender_relax_loss_alpha', type=float, default=0.5,
                                help='Alpha parameter for Relax Loss defense')
        # Adversarial Regularization parameters for defender model
        parser.add_argument('--defender_adv_reg_lambda', type=float, default=1.0,
                                help='Lambda parameter for Adversarial Regularization defense')
        parser.add_argument('--defender_adv_reg_shadow_attacker_model_layers', type=int, nargs="+", default=[64, 32],
                                help='List of hidden layer sizes for the shadow attacker model in Adversarial Regularization defense')
        parser.add_argument('--defender_adv_reg_shadow_attacker_k', type=int, default=1,
                                help='Number of shadow attacker steps per defender step in Adversarial Regularization defense')
        # Mixup paremeters for defender model
        parser.add_argument('--defender_mixup_alpha', type=float, default=1.0,
                                help='Alpha parameter for Mixup defense')
        # HAMP parameters for defender model
        parser.add_argument('--defender_hamp_gamma', type=float, default=0.5,
                                help='gamma parameter for HAMP defense')
        parser.add_argument('--defender_hamp_alpha', type=float, default=0.001,
                                help='Alpha parameter for HAMP defense')
        # Selena parameters for defender model
        parser.add_argument('--defender_selena_K', type=int, default=25,
                                help='Number of models K for Selena defense')
        parser.add_argument('--defender_selena_L', type=int, default=10,
                                help='Number of exclusions L per sample for Selena defense')
        # MIST parameters for defender model
        parser.add_argument('--defender_mist_num_submodels', type=int, default=5,
                                help='Number of submodels to train for MIST defense')
        parser.add_argument('--defender_mist_split_method', type=str, default='random',
                                help='Data split method for MIST defense', choices=['random', 'stratified'])
        parser.add_argument('--defender_mist_submodel_epochs', type=int, default=5,
                                help='Number of epochs to train each submodel for MIST defense')
        parser.add_argument('--defender_mist_lambda', type=float, default=4,
                                help='Lambda parameter for MIST defense')
        # Weighted Smoothing parameters for defender model
        parser.add_argument('--defender_weighted_smoothing_sigma_noise', type=float, default=0.1,
                                help='Standard deviation of the gaussian noise to be added in Weighted Smoothing defense')
        parser.add_argument('--defender_weighted_smoothing_warmup_epochs', type=int, default=10,
                                help='Number of warmup epochs to train the model without weighted smoothing in Weighted Smoothing defense')
        # Purifier parameters for defender model
        parser.add_argument('--defender_purifier_reformer_latent_dim', type=int, default=16,
                                help='Latent dimension for the reformer model in Purifier defense')
        parser.add_argument('--defender_purifier_reformer_hidden_dim', type=int, default=128,
                                help='Hidden dimension for the reformer model in Purifier defense')
        parser.add_argument('--defender_purifier_reformer_epochs', type=int, default=20,
                                help='Number of epochs to train the reformer model in Purifier defense')
        parser.add_argument('--defender_purifier_reformer_lr', type=float, default=0.01,
                                help='Learning rate to train the reformer model in Purifier defense')
        parser.add_argument('--defender_purifier_reformer_batch_size', type=int, default=256,
                                help='Batch size to train the reformer model in Purifier defense')
        parser.add_argument('--defender_purifier_reformer_lambda', type=float, default=1.0,
                                help='Lambda parameter to weight the reformer loss in Purifier defense')
        parser.add_argument('--defender_purifier_pindex_size', type=int, default=1000,
                                help='Number of samples to use in the Pindex for the Purifier defense')
        parser.add_argument('--defender_purifier_swap_threshold', type=float, default=0.01,
                                help='Threshold for label swapping in the Purifier defense')
        # MMD parameters for defender model
        parser.add_argument('--defender_mmd_lambda', type=float, default=1.0,
                                help='Lambda parameter for MMD defense')
        # LDL parameters for defender model
        parser.add_argument('--defender_ldl_n_queries', type=int, default=100,
                                help='Number of queries to train the LDL defense layer')
        parser.add_argument('--defender_ldl_noise_scale', type=float, default=0.1,
                                help='Standard deviation of the noise to be added in the LDL defense layer')
        # Data Augmentation parameters for defender model
        parser.add_argument('--defender_augment_horizontal_flip', action="store_true", default=False,
                                help="whether to apply horizontal flip augmentation for the Data Augmentation defense",)
        parser.add_argument('--defender_augment_rotation', type=float, default=10,
                                help='degree of random rotation augmentation for the Data Augmentation defense')
        parser.add_argument('--defender_augment_random_crop', type=int, default=32,
                                help="whether to apply random crop augmentation for the Data Augmentation defense",)
        parser.add_argument('--defender_augment_jitter_brightness', type=float, default=0.2,
                                help='brightness jitter factor for the Color Jitter augmentation in the Data Augmentation defense')
        parser.add_argument('--defender_augment_jitter_hue', type=float, default=0.2,
                                help='hue jitter factor for the Color Jitter augmentation in the Data Augmentation defense')
        parser.add_argument('--defender_augment_perspective_distortion_scale', type=float, default=0.2,
                                help='distortion scale for the Random Perspective augmentation in the Data Augmentation defense')
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
                                        'pmia_loss', 'pmia_confidence', 'pmia_entropy',
                                        'sba', 'sba_hopskipjump', 'sba_hsj', 'sba_hopskip', 'sba_hop', 'sba_qeba', 'sba_qeba-spatial', 'sba_qeba-dct', 'sba_qeba-pca', 'sba_qeba-custom',
                                        'uba', 'uba_hopskipjump', 'uba_hsj', 'uba_hopskip', 'uba_hop', 'uba_qeba', 'uba_qeba-spatial', 'uba_qeba-dct', 'uba_qeba-pca', 'uba_qeba-custom',
                                        'noise_robust', 'noise_robustness', 'noise_rob', 'nr',
                                        'transfer_loss', 'transfer_confidence', 'transfer_entropy',
                                        'oslo', 'oslo_difgsm', 'oslo_mifgsm', 'oslo_tifgsm', 'oslo_tmifgsm',
                                        'dh', 'dh_white', 'dh_black', 'dh_random', 'dh-attack', 'dh-attack_white', 'dh-attack_black', 'dh-attack_random',
                                        'online_yoqo', 'offline_yoqo', 'on_yoqo', 'off_yoqo', 'yoqo',
                                        ])
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
        # Boundary MIA parameters
        parser.add_argument('--bound_n_queries', type=int, default=1000,
                                help='Number of queries to train the attack regressor in the Boundary MIA attack')
        parser.add_argument('--bound_norm', type=str, default='l2',
                                help='Norm to use for the hopskip steps in the Boundary MIA attack')
        parser.add_argument('--bound_qeba_reduction_factor', type=int, default=8,
                                help='Reduction factor for QEBA (spatial/dct variants)')
        parser.add_argument('--bound_quantile', type=float, default=0.5,
                                help='Quantile to use for the unsupervised variant of the Boundary MIA attack')
        # Noise Robustness MIA parameters
        parser.add_argument('--noise_robust_n_queries', type=int, default=5000,
                                help='Number of noisy copies to create for each sample in the Noise Robustness MIA attack')
        parser.add_argument('--noise_robust_sigmas', type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25], #[0.1, 0.2, 0.3, 0.4, 0.5], #[0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2],
                                help='Standard deviations of the gaussian noise to be added in the Noise Robustness MIA attack')
        # OSLO MIA parameters
        parser.add_argument('--oslo_n_models', type=int, default=10,
                                help='Number of shadow models to be trained for the OSLO MIA attack')
        parser.add_argument('--oslo_same_arch', action='store_true', default=False,
                                help='Whether to use the same architecture for all shadow models or not')
        parser.add_argument('--oslo_source_models_ratio', type=float, default=0.75,
                                help='Ratio of shadow models to use as source models for the attack (the rest will be used as validation models)')
        parser.add_argument('--oslo_K', type=int, default=10,
                                help='Number of attack sub-procedures for the OSLO MIA attack')
        parser.add_argument('--oslo_N', type=int, default=1000,
                                help='Number of attack iterations per sub-procedure for the OSLO MIA attack')
        parser.add_argument('--oslo_max_epsilon', type=float, default=4/255,
                                help='Maximum perturbation for the OSLO MIA attack')
        parser.add_argument('--oslo_threshold', type=float, default=0.01,
                                help='Decision threshold for the OSLO MIA attack')
        # DHAttack MIA parameters
        parser.add_argument('--dh_n_models', type=int, default=10,
                                help='Number of shadow models to be trained for the DHAttack MIA attack')
        parser.add_argument('--dh_n_queries', type=int, default=1000,
                                help='Number of queries to train the attack regressor in the DHAttack MIA attack')
        # YOQO MIA parameters
        parser.add_argument('--yoqo_alpha', type=float, default=2,
                                help='Alpha parameter for the YOQO MIA attack')
        parser.add_argument('--yoqo_gamma', type=float, default=5,
                                help='Gamma parameter for the YOQO MIA attack')
        parser.add_argument('--yoqo_adv_opt_max_iter', type=int, default=100,
                                help='Maximum number of iterations for the adversarial optimization in the YOQO MIA attack')
        parser.add_argument('--yoqo_adv_opt_lr', type=float, default=0.01,
                                help='Learning rate for the adversarial optimization in the YOQO MIA attack')
        parser.add_argument('--yoqo_adv_opt_loss_threshold', type=float, default=6,
                                help='Loss threshold for the adversarial optimization in the YOQO MIA attack')

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
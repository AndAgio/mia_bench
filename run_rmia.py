import os
from src.utils.settings import gather_settings
from src.utils.configs import TrainConfigs, LogConfigs, ModelConfigs, DatasetConfigs, AuditingDataConfigs, ShadowDataConfigs
from src.utils.log import get_logger_from_configs
from src.mia.victim import Victim
from src.mia.rmia import RMIA


def main():
    # TODO: Make main adaptable to multiple MIAs.
    # Issue URL: https://github.com/AndAgio/mia_bench/issues/7
    # assignees: AndAgio.
    settings = gather_settings()

    exp_code = f'{settings.attack_mode}_{settings.dataset}_victim_{settings.victim_model}_{settings.victim_optimizer}_{settings.victim_lr}_{settings.victim_lr_sched}_{settings.victim_seed}_attacker_{settings.att_model}_{settings.att_optimizer}_{settings.att_lr}_{settings.att_lr_sched}_{settings.att_seed}'
    exp_log_folder = settings.log_folder/f'{exp_code}'
    exp_ckpts_folder = settings.models_folder/f'{exp_code}'
    exp_resume_ckpts_folder = settings.resume_ckpts_folder/f'{exp_code}'

    dataset_configs = DatasetConfigs(name=settings.dataset,
                                    data_folder=settings.datasets_folder,
                                    data_augmentation=settings.data_augmentation,)
    victim_log_configs = LogConfigs(name='victim',
                                    log_folder=exp_log_folder,
                                    log_mode='smart')
    victim_model_configs = ModelConfigs(model_name=settings.victim_model,
                                        im_channels=dataset_configs.im_channels,
                                        num_classes=dataset_configs.num_classes,
                                        im_size=dataset_configs.im_size,)
    victim_train_configs = TrainConfigs(optimizer=settings.victim_optimizer,
                                        lr=settings.victim_lr,
                                        epochs=settings.victim_epochs,
                                        batch_size=settings.victim_batch_size,
                                        loss=settings.victim_loss,
                                        lr_sched=settings.victim_lr_sched,
                                        device=settings.device,
                                        seed=settings.victim_seed,
                                        distributed=settings.distributed,
                                        ckpts_folder=exp_ckpts_folder,
                                        resume_ckpts_folder=exp_resume_ckpts_folder)

    victim = Victim(dataset_configs=dataset_configs,
                    model_configs=victim_model_configs,
                    logger=get_logger_from_configs(victim_log_configs))
    victim_model = victim.train_model(train_configs=victim_train_configs,)
    

    attacker_log_configs = LogConfigs(name='attacker',
                                    log_folder=exp_log_folder,
                                    log_mode='smart')
    attacker_model_configs = ModelConfigs(model_name=settings.att_model,
                                        im_channels=dataset_configs.im_channels,
                                        num_classes=dataset_configs.num_classes,
                                        im_size=dataset_configs.im_size,)
    attacker_train_configs = TrainConfigs(optimizer=settings.att_optimizer,
                                        lr=settings.att_lr,
                                        epochs=settings.att_epochs,
                                        batch_size=settings.att_batch_size,
                                        loss=settings.att_loss,
                                        lr_sched=settings.att_lr_sched,
                                        device=settings.device,
                                        seed=settings.att_seed,
                                        distributed=settings.distributed,
                                        ckpts_folder=exp_ckpts_folder,
                                        resume_ckpts_folder=exp_resume_ckpts_folder)
    attacker_audit_configs = AuditingDataConfigs(n_auditing_samples=settings.n_auditing_samples,
                                                in_perc=settings.audit_in_perc,
                                                seed=settings.att_seed)
    attacker_shadow_configs = ShadowDataConfigs(n_shadow_datasets=settings.n_shadows,
                                                n_samples_per_dataset=settings.n_samples_per_shadow_dataset,
                                                mode='offline',
                                                test_perc=settings.shadow_test_perc,
                                                seed=settings.att_seed)

    attacker = RMIA(victim_model=victim_model,
                    victim_dataset=victim.get_dataset(),
                    audit_configs=attacker_audit_configs,
                    shadow_configs=attacker_shadow_configs,
                    model_configs=attacker_model_configs,
                    logger=get_logger_from_configs(attacker_log_configs))
    attacker.optimize(train_config=attacker_train_configs)
    scores = attacker.measure_effectiveness(alpha=settings.rmia_alphas,
                                            random_pop_size=settings.random_population_size,
                                            device=attacker_train_configs.device)
    # attacker.measure_effectiveness()
    # auc_score, tpr, fpr, roc = attacker.get_stats(scores=scores)
    for alpha, scores_dict in scores.items():
        print(f'Alpha = {alpha} -> AUC = {scores_dict['auc']}')

    # TODO: Find better way to store and deal with experiment results.
    # Issue URL: https://github.com/AndAgio/mia_bench/issues/6
    # assignees: AndAgio.
    exp_results_folder = settings.out_folder/f'{exp_code}'
    os.makedirs(exp_results_folder, exist_ok=True)
    with open(os.path.join(exp_results_folder, 'results.txt'), 'w') as file:
        for alpha, scores_dict in scores.items():
            file.write(f'Alpha = {alpha} -> AUC: {scores_dict['auc']}\n')
            file.write(f'Alpha = {alpha} -> TPR: {scores_dict['tpr']}\n')
            file.write(f'Alpha = {alpha} -> FPR: {scores_dict['fpr']}\n')
            file.write(f'Alpha = {alpha} -> ROC: {scores_dict['roc']}\n')
            file.write('\n')


if __name__ == '__main__':
    main()
from src.utils.settings import gather_settings
from src.trainer.utils import TrainConfigs
from src.mia.victim import Victim


def main():

    settings = gather_settings()

    victim = Victim(dataset=settings.dataset,
                    model_name=settings.victim_model,
                    datasets_folder=settings.datasets_folder, 
                    data_augmentation=settings.data_augmentation)
    victim_train_configs = TrainConfigs(optimizer=settings.optimizer,
                                        lr=settings.lr,
                                        epochs=settings.epochs,
                                        batch_size=settings.batch_size,
                                        loss=settings.loss,
                                        lr_sched=settings.lr_sched,
                                        device=settings.device,
                                        seed=settings.seed,
                                        distributed=settings.distributed,
                                        ckpts_folder=settings.models_folder,
                                        resume_ckpts_folder=settings.resume_ckpts_folder)
    victim.train_model(train_configs=victim_train_configs,
                        log_folder=settings.log_folder,
                        logging_mode='smart')

    # attacker = Mia()
    # auditing_dataset = attacker.build_auditing_dataset(victim_dataset)
    # attacker.optimize()
    # attacker.measure_effectiveness(auditing_dataset)


if __name__ == '__main__':
    main()
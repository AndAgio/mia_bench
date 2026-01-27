from src.utils.settings import gather_settings, setup_configs_and_folder_from_settings
from src.mia.victim import Victim


def main():
    settings = gather_settings()
    experiment_configs, _ = setup_configs_and_folder_from_settings(settings)

    victim = Victim(victim_configs=experiment_configs.victim)
    _ = victim.train_model(train_configs=experiment_configs.victim.train,)


if __name__ == '__main__':
    main()
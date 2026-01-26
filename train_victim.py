from src.utils.settings import gather_settings, setup_configs_and_folder_from_settings
from src.mia.defenses import get_defender_class


def main():
    settings = gather_settings()
    experiment_configs, exp_out_folder = setup_configs_and_folder_from_settings(settings)

    defender_class = get_defender_class(settings.defense_mode)
    defender = defender_class(defender_configs=experiment_configs.defender)
    defender_model, defender_stats = defender.optimize(train_configs=experiment_configs.defender.train, return_stats=True)


if __name__ == '__main__':
    main()
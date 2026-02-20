from src.utils.settings import gather_settings, setup_configs_and_folder_from_settings
from src.mia.defenses import get_defender_class


def get_device(dev_str: str = 'cpu'):
    import torch
    # Set appropriate devices
    if torch.cuda.is_available() and dev_str != 'cpu':
        dev_str = 'cuda:{}'.format(dev_str)
        device = torch.device(dev_str)
    elif torch.backends.mps.is_available() and dev_str != 'cpu':
        dev_str = 'mps'
        device = torch.device(dev_str)
    else:
        device = torch.device('cpu')
    return device


def main():
    settings = gather_settings()
    experiment_configs, exp_out_folder = setup_configs_and_folder_from_settings(settings)

    print("\n\n================= Device Configurations =================")
    print(f"Using device: {experiment_configs.defender.train.device}")
    print(f"Device name: {get_device(experiment_configs.defender.train.device)}")
    print("=========================================================\n\n")

    defender_class = get_defender_class(settings.defense_mode)
    defender = defender_class(defender_configs=experiment_configs.defender)
    trained_defender_model, defender_stats = defender.train_model(train_configs=experiment_configs.defender.train, return_stats=True)
    defender.defend_model(device=experiment_configs.defender.train.device)
    defender_model = defender.get_defended_model()


if __name__ == '__main__':
    main()
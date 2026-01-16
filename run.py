import json
import os
from pathlib import PosixPath
from src.utils.settings import gather_settings
from src.utils.configs import get_hash_from_settings, generate_configs_from_settings
from src.mia.victim import Victim
from src.mia import get_attacker_class


def main():
    settings = gather_settings()
    exp_hash = get_hash_from_settings(settings)
    exp_out_folder = settings.out_folder/exp_hash
    os.makedirs(exp_out_folder, exist_ok=True)
    settings_file = exp_out_folder/'settings.json'
    with open(settings_file, 'w') as f:
        settings_dict = {k: str(v) if isinstance(v, PosixPath) else v for k, v in vars(settings).items()}
        json.dump(settings_dict, f, indent=4)
        print(f'Experiment settings saved to {settings_file}')
    experiment_configs = generate_configs_from_settings(settings, exp_hash)

    # TODO: Separate output folders for victim and attacker depending on their own configs, so that victim can be only one for all attackers.
    # assignees: AndAgio.
    victim = Victim(victim_configs=experiment_configs.victim)
    victim_model = victim.train_model(train_configs=experiment_configs.victim.train,)

    attacker_class = get_attacker_class(settings.attack_mode)
    attacker = attacker_class(victim_model=victim_model,
                            victim_dataset=victim.get_dataset(),
                            attacker_configs=experiment_configs.attacker,
                            exp_hash=exp_hash)
    attacker.optimize(train_config=experiment_configs.attacker.train)
    attacker.measure_effectiveness(device=experiment_configs.attacker.train.device)

    best_auc_params, best_auc = attacker.get_best_result('auc', mode='max')
    print('Best AUC was obtained for parameters: {} and was {}'.format(best_auc_params, best_auc))
    # print(attacker.summarize_results())

    exp_results_folder = exp_out_folder/'results'
    os.makedirs(exp_results_folder, exist_ok=True)
    attacker.save_results_to_json(os.path.join(exp_results_folder, 'results.json'))


if __name__ == '__main__':
    main()
import json
import os
from pathlib import PosixPath
from src.utils.settings import gather_settings, setup_configs_and_folder_from_settings
from src.utils.configs import get_hash_from_settings, generate_configs_from_settings
from src.mia.victim import Victim
from src.mia import get_attacker_class


def main():
    settings = gather_settings()
    experiment_configs, exp_out_folder = setup_configs_and_folder_from_settings(settings)

    victim = Victim(victim_configs=experiment_configs.victim)
    victim_model, victim_stats = victim.train_model(train_configs=experiment_configs.victim.train, return_stats=True)

    attacker_class = get_attacker_class(settings.attack_mode)
    attacker = attacker_class(victim_model=victim_model,
                            victim_dataset=victim.get_dataset(),
                            attacker_configs=experiment_configs.attacker)
    attacker.optimize(train_config=experiment_configs.attacker.train)
    attacker.measure_effectiveness(device=experiment_configs.attacker.train.device)

    best_auc_params, best_auc = attacker.get_best_result('auc', mode='max')
    # TODO: Refactor stats tracking to make it easier to get victim training stats.
    # assignees: AndAgio
    best_epoch, best_acc = victim_stats.get_best()
    print(f"Victim stats: best {victim_stats.stage_to_track_best} accuracy = {best_acc}")
    print('Best AUC was obtained for parameters: {} and was {}'.format(best_auc_params, best_auc))
    # print(attacker.summarize_results())

    exp_results_folder = exp_out_folder/'results'
    os.makedirs(exp_results_folder, exist_ok=True)
    attacker.save_results_to_json(os.path.join(exp_results_folder, 'results.json'))


if __name__ == '__main__':
    main()
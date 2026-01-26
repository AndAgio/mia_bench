import json
import os
from pathlib import PosixPath
from src.utils.settings import gather_settings, setup_configs_and_folder_from_settings
from src.utils.configs import get_hash_from_settings, generate_configs_from_settings
# from mia.defenses.vanilla import Victim
from src.mia.defenses import get_defender_class
from src.mia.attacks import get_attacker_class


def main():
    settings = gather_settings()
    experiment_configs, exp_out_folder = setup_configs_and_folder_from_settings(settings)

    defender_class = get_defender_class(settings.defense_mode)
    defender = defender_class(defender_configs=experiment_configs.defender)
    defender_model, defender_stats = defender.optimize(train_configs=experiment_configs.defender.train, return_stats=True)

    # victim = Victim(victim_configs=experiment_configs.victim)
    # victim_model, victim_stats = victim.train_model(train_configs=experiment_configs.victim.train, return_stats=True)

    attacker_class = get_attacker_class(settings.attack_mode)
    attacker = attacker_class(defender_model=defender_model,
                            defender_dataset=defender.get_dataset(),
                            attacker_configs=experiment_configs.attacker)
    attacker.optimize(train_config=experiment_configs.attacker.train)
    attacker.measure_effectiveness(device=experiment_configs.attacker.train.device)

    best_auc_params, best_auc = attacker.get_best_result('auc', mode='max')
    # TODO: Refactor stats tracking to make it easier to get victim training stats.
    # Issue URL: https://github.com/AndAgio/mia_bench/issues/20
    # assignees: AndAgio
    best_epoch, best_acc = defender_stats.get_best()
    print(f"Defender stats: best {defender_stats.stage_to_track_best} accuracy = {best_acc}")
    print('Best AUC was obtained for parameters: {} and was {}'.format(best_auc_params, best_auc))
    # print(attacker.summarize_results())

    exp_results_folder = exp_out_folder/'results'
    os.makedirs(exp_results_folder, exist_ok=True)
    attacker.save_results_to_json(os.path.join(exp_results_folder, 'results.json'))


if __name__ == '__main__':
    main()
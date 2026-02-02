import json
import os
from pathlib import PosixPath
from src.utils.settings import gather_settings, setup_configs_and_folder_from_settings
from src.utils.configs import get_hash_from_settings, generate_configs_from_settings
from src.mia.victim import Victim
from src.mia import get_attacker_class
import src.trainer.prop_noise_obfs as prop_noise_obfs
from src.trainer import train_manager

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
    # Issue URL: https://github.com/AndAgio/mia_bench/issues/20
    # assignees: AndAgio
    best_epoch, best_acc = victim_stats.get_best()
    #print(f"Victim stats: best {victim_stats.stage_to_track_best} accuracy = {best_acc}")
    #print('Best AUC was obtained for parameters: {} and was {}'.format(best_auc_params, best_auc))
    # print(attacker.summarize_results())
    best_epoch, best_acc = victim_stats.get_best()
    #print(f"Victim stats: best {victim_stats.stage_to_track_best} accuracy = {best_acc}")

    
    exp_results_folder = exp_out_folder/'results'
    os.makedirs(exp_results_folder, exist_ok=True)
    attacker.save_results_to_json(os.path.join(exp_results_folder, 'results.json'))
    print("\n")
    print("\n")
    if (experiment_configs.victim.train.metric_config.use_metric == True):
        final_epsilon = train_manager.final_epsilon
        final_delta = train_manager.final_delta
        if (final_epsilon==-1 or final_delta ==-1):
            print("No feasible integer λ found (within caps). ")
            print("Try increasing lam_max_hard, or change (b,d,q,T).")
            print("In other words, theorem 2.1. cannot be applied with this b and d!")

        print("***** Runned METRIC PRIVACY defence ****")
        print(f"Final ε = {final_epsilon:.3f}, δ = {final_delta:.2e}")
        print("*******************")
    else:
        print("Vanilla - without metric privacy defence")
    print("-----------------")
    print(f"Model Accuracy:", best_acc, "| Attack AUC:",float(best_auc))
    print("-----------------")
if __name__ == '__main__':
    main()
import json
import os
import resource
import sys
import time
from dataclasses import asdict
from pathlib import PosixPath
import torch
from torch.utils.data import DataLoader
from src.utils.settings import gather_settings, setup_configs_and_folder_from_settings
from src.utils.configs import get_hash_from_settings, generate_configs_from_settings
# from mia.defenses.vanilla import Victim
from src.mia.defenses import get_defender_class
from src.mia.attacks import get_attacker_class


def resolve_device(device):
    if isinstance(device, str):
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device(f'cuda:{device}')
        elif torch.backends.mps.is_available() and device != 'cpu':
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    return device


def evaluate_model_performance(model, dataset, device, batch_size: int) -> dict:
    """Evaluate accuracy/loss for the final, possibly post-hoc defended, model."""
    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, targets, _, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            outputs = model(inputs.to(device))
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            predictions = torch.argmax(outputs, dim=1).cpu()
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
            outputs_cpu = outputs.detach().cpu()
            if (torch.all(outputs_cpu >= 0)
                    and torch.allclose(outputs_cpu.sum(dim=1), torch.ones(outputs_cpu.shape[0]),
                                       atol=1e-4, rtol=1e-4)):
                probabilities = outputs_cpu.clamp_min(1e-12)
                loss_sum += float(-torch.log(probabilities[torch.arange(len(targets)), targets]).sum())
            else:
                loss_sum += float(torch.nn.functional.cross_entropy(
                    outputs_cpu, targets, reduction='sum'))
    return {
        'accuracy': correct / total if total else 0.0,
        'loss': loss_sum / total if total else 0.0,
        'n_samples': total,
    }


def process_peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == 'darwin' else 1024
    return float(peak / divisor)


def aggregation_train_configuration(config) -> dict:
    """Keep result-affecting training settings while excluding seed/path/runtime state."""
    raw = asdict(config)
    for field in ('seed', 'device', 'distributed', 'resume',
                  'ckpts_folder', 'resume_ckpts_folder'):
        raw.pop(field, None)

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return getattr(value, '__name__', str(value))
    return normalize(raw)


def main():
    settings = gather_settings()
    experiment_configs, exp_out_folder = setup_configs_and_folder_from_settings(settings)

    defender_class = get_defender_class(settings.defender_mode)
    defender = defender_class(defender_configs=experiment_configs.defender)
    defender_train_start = time.perf_counter()
    trained_defender_model, defender_stats = defender.train_model(train_configs=experiment_configs.defender.train, return_stats=True)
    defender_training_seconds = time.perf_counter() - defender_train_start
    defense_start = time.perf_counter()
    defender.defend_model(device=experiment_configs.defender.train.device)
    defense_application_seconds = time.perf_counter() - defense_start
    defender_model = defender.get_defended_model()
    # defender_model, defender_stats = defender.optimize(train_configs=experiment_configs.defender.train, return_stats=True)

    # victim = Victim(victim_configs=experiment_configs.victim)
    # victim_model, victim_stats = victim.train_model(train_configs=experiment_configs.victim.train, return_stats=True)

    attacker_class = get_attacker_class(settings.attacker_mode)
    attacker = attacker_class(defender_model=defender_model,
                            attacker_configs=experiment_configs.attacker)
    attack_optimization_start = time.perf_counter()
    attacker.optimize(train_config=experiment_configs.attacker.train)
    attack_optimization_seconds = time.perf_counter() - attack_optimization_start

    # Compute class/correctness slices outside the query counter: this is evaluation,
    # not part of the attack's target-model query budget.
    attacker.prepare_audit_analysis(device=experiment_configs.attacker.train.device)
    target_query_count = 0
    def count_target_queries(_module, inputs):
        nonlocal target_query_count
        if inputs and hasattr(inputs[0], 'shape') and len(inputs[0].shape) > 0:
            target_query_count += int(inputs[0].shape[0])
        else:
            target_query_count += 1

    query_hook = defender_model.register_forward_pre_hook(count_target_queries)
    attack_evaluation_start = time.perf_counter()
    try:
        attacker.measure_effectiveness(device=experiment_configs.attacker.train.device)
    finally:
        query_hook.remove()
    attack_evaluation_seconds = time.perf_counter() - attack_evaluation_start

    best_auc_params, best_auc = attacker.get_best_result('auc', mode='max')
    # TODO: Refactor stats tracking to make it easier to get victim training stats.
    # Issue URL: https://github.com/AndAgio/mia_bench/issues/20
    # assignees: AndAgio
    best_epoch, best_acc = defender_stats.get_best()
    print(f"Defender stats: best {defender_stats.stage_to_track_best} accuracy = {best_acc}")
    print('Best AUC was obtained for parameters: {} and was {}'.format(best_auc_params, best_auc))
    train_performance = evaluate_model_performance(
        model=defender_model,
        dataset=defender.get_dataset().get('train'),
        device=experiment_configs.defender.train.device,
        batch_size=experiment_configs.defender.train.batch_size,
    )
    test_performance = evaluate_model_performance(
        model=defender_model,
        dataset=defender.get_dataset().get('test'),
        device=experiment_configs.defender.train.device,
        batch_size=experiment_configs.defender.train.batch_size,
    )
    n_audit_samples = len(attacker.audit_manager.get_all_ids())
    shadow_manager = getattr(attacker, 'shadow_manager', None)
    shadow_model_count = (shadow_manager.get_n_models()
                          if shadow_manager is not None
                          and getattr(shadow_manager, 'shadow_models', None) is not None else 0)
    shadow_samples_per_dataset = (
        int(shadow_manager.shadow_data.n_samples_per_dataset)
        if shadow_manager is not None and getattr(shadow_manager, 'shadow_data', None) is not None
        else None
    )
    attack_configs = experiment_configs.attacker.attack
    configured_queries = getattr(attack_configs, 'n_queries', None)
    if configured_queries is None and hasattr(attack_configs, 'boundary'):
        configured_queries = attack_configs.boundary.n_queries
    if getattr(attack_configs, 'strategy', None) == 'robust':
        reference_population_size = int(attack_configs.random_pop_size)
    elif getattr(attack_configs, 'strategy', None) == 'attack_p':
        reference_population_size = shadow_samples_per_dataset
    else:
        reference_population_size = None
    device = resolve_device(experiment_configs.attacker.train.device)
    cuda_peak_memory_mb = (float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                           if device.type == 'cuda' else None)
    attacker.add_run_metrics(
        attacker_mode=settings.attacker_mode,
        defender_mode=settings.defender_mode,
        defender_model_name=experiment_configs.defender.model.model_name,
        attacker_configuration=asdict(experiment_configs.attacker.attack),
        defense_configuration=asdict(experiment_configs.defender.defense),
        defender_training_configuration=aggregation_train_configuration(
            experiment_configs.defender.train),
        attacker_training_configuration=aggregation_train_configuration(
            experiment_configs.attacker.train),
        dataset_name=experiment_configs.defender.dataset.base.name,
        defender_split=float(experiment_configs.defender.dataset.base.def_split),
        attacker_split=float(experiment_configs.defender.dataset.base.att_split),
        audit_in_percentage=float(experiment_configs.attacker.dataset.auditing.in_perc),
        configured_audit_samples=int(experiment_configs.attacker.dataset.auditing.n_auditing_samples),
        configured_shadow_datasets=int(experiment_configs.attacker.dataset.shadow.n_shadow_datasets),
        seed=int(experiment_configs.defender.dataset.base.seed),
        model_accuracy=float(test_performance['accuracy']),
        model_accuracy_split='test',
        model_train_accuracy=float(train_performance['accuracy']),
        model_test_accuracy=float(test_performance['accuracy']),
        model_train_loss=float(train_performance['loss']),
        model_test_loss=float(test_performance['loss']),
        accuracy_generalization_gap=float(train_performance['accuracy'] - test_performance['accuracy']),
        loss_generalization_gap=float(test_performance['loss'] - train_performance['loss']),
        training_best_metric=float(best_acc),
        training_best_metric_name=defender_stats.metric_to_track_best,
        training_best_stage=defender_stats.stage_to_track_best,
        training_best_epoch=int(best_epoch),
        defender_training_seconds=float(defender_training_seconds),
        defense_application_seconds=float(defense_application_seconds),
        attack_optimization_seconds=float(attack_optimization_seconds),
        attack_evaluation_seconds=float(attack_evaluation_seconds),
        target_model_queries_total=int(target_query_count),
        target_model_queries_mean_per_audit_sample=float(target_query_count / n_audit_samples),
        audit_samples_per_evaluation_second=float(n_audit_samples / attack_evaluation_seconds),
        configured_max_queries_per_sample=(int(configured_queries)
                                           if configured_queries is not None else None),
        shadow_model_count=int(shadow_model_count),
        shadow_samples_per_dataset=shadow_samples_per_dataset,
        reference_population_size=reference_population_size,
        process_peak_memory_mb=process_peak_memory_mb(),
        cuda_peak_memory_mb=cuda_peak_memory_mb,
    )
    # print(attacker.summarize_results())

    exp_results_folder = exp_out_folder/'results'
    os.makedirs(exp_results_folder, exist_ok=True)
    attacker.save_results_to_json(os.path.join(exp_results_folder, 'results.json'))
    print(f"Results written to {exp_results_folder} (summary.csv, summary.json, membership_predictions.jsonl, results.json)")


if __name__ == '__main__':
    main()

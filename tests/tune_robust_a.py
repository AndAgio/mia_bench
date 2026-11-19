#!/usr/bin/env python3
"""Find RobustMIA's default a for a dataset with the paper's procedure (App. B.2.2), as the authors' code implements it
(tune_offline_a in privacy_meter's attacks.py). Run once per dataset, offline: the attack itself never tunes a.

The script trains (or loads from the cache) offline RobustMIA's shadow models with the given run.py options. Each
shadow model is then attacked in turn, with the other shadow models as references, for a = 0, 0.1, ..., 1, on up to
500 of its members and 500 non-members, against the population the attack would draw. Only the attacker's own models
and data are used: no target model is trained or queried. The a with the best AUC, averaged over the attacked models,
goes into DEFAULT_A in src/mia/attacks/black_box/robust.py.

Usage: python tests/tune_robust_a.py --dataset svhn --n_shadows 2 --device mps [other run.py options]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mia.attacks.black_box.robust import RobustMIA, offline_p_x, rmia_scores
from src.utils.configs import get_hash_from_settings
from src.utils.settings import build_parser, setup_configs_and_folder_from_settings

# The values of a tried, as in the authors' code.
TUNING_A_GRID = [round(0.1 * step, 1) for step in range(11)]
# At most this many members and non-members of each attacked shadow model.
TUNING_SAMPLES = 500
# Label of the random stream that picks them, derived from the seed (like the streams of shadow_data_manager.py).
TUNING_STREAM = 3


def tuning_aucs(probs, sample_ids, is_member, population_probs, population_ids, maps, attacked, gamma):
    """AUC of offline RobustMIA against the shadow model in column `attacked`, with the other shadow models as
    references, for each a in TUNING_A_GRID.

    probs and population_probs hold the probabilities of the tuning samples and of the population under every shadow
    model, one column each; maps holds each shadow model's training samples. As in the attack, Pr(x) and Pr(z) both
    average the references not trained on the sample, through a. The population keeps only non-members of the attacked
    model, as the attack's population holds non-members of the target, with at least one reference not trained on them.
    """
    references = [column for column in range(probs.shape[1]) if column != attacked]
    reference_sets = [set(maps[column]) for column in references]

    def out_of(samples):
        """Whether each reference model was not trained on each sample."""
        return np.array([[sample not in members for members in reference_sets] for sample in samples],
                        dtype=bool).reshape(len(samples), len(references))

    def p_out(sample_probs, out):
        return (sample_probs[:, references] * out).sum(axis=1) / np.maximum(out.sum(axis=1), 1)
    out_x, out_z = out_of(sample_ids), out_of(population_ids)
    keep = out_x.any(axis=1)
    attacked_members = set(maps[attacked])
    z_keep = np.array([z not in attacked_members for z in population_ids]) & out_z.any(axis=1)
    p_x_out, p_z_out = p_out(probs, out_x)[keep], p_out(population_probs, out_z)[z_keep]
    is_member = np.asarray(is_member, dtype=bool)[keep]
    return [roc_auc_score(is_member, rmia_scores(probs[keep, attacked] / (offline_p_x(p_x_out, a) + 1e-15),
                                                 population_probs[z_keep, attacked] / (offline_p_x(p_z_out, a) + 1e-15),
                                                 gamma))
            for a in TUNING_A_GRID]


def main(argv=None):
    """Train or load the shadow models, attack each one in turn and report the AUC of every a."""
    parser = build_parser()
    settings = parser.parse_args(argv)
    settings.attacker_mode = 'offline_robust'
    if settings.n_shadows < 2:
        parser.error('the tuning attacks one shadow model with the others as references: '
                     '--n_shadows must be at least 2')
    configs, _ = setup_configs_and_folder_from_settings(settings)
    # The tuning does not score with a, so a placeholder lets datasets without a default build the attack. It is set in
    # the configs, not the settings, so the attacker hash, and with it the cached shadow models, is a real run's.
    if configs.attacker.attack.a is None:
        configs.attacker.attack.a = 0.5
    attack = RobustMIA(defender_model=None, attacker_configs=configs.attacker)
    device = configs.attacker.train.device
    attack.optimize(train_config=configs.attacker.train)

    shadow_data = attack.shadow_manager.shadow_data
    model_ids = attack.shadow_manager.get_all_model_indeces()
    maps = [shadow_data.get_map(model) for model in model_ids]
    rng = np.random.default_rng([settings.seed, TUNING_STREAM])
    rotations = []
    for column, model in enumerate(model_ids):
        outside = shadow_data.indices_outside_shadow_dataset(model)
        members = rng.choice(maps[column], min(TUNING_SAMPLES, len(maps[column])), replace=False).tolist()
        non_members = rng.choice(outside, min(TUNING_SAMPLES, len(outside)), replace=False).tolist()
        rotations.append((members, non_members))
    sample_ids = sorted({sample for members, non_members in rotations for sample in members + non_members})
    samples = attack.shadow_manager.get_random_population(indices=sample_ids, labels='original')
    probs = attack.compute_p_x_theta(audit_dataset=samples, models_for_sample=[model_ids for _ in sample_ids],
                                     device=device)
    row = {sample: index for index, sample in enumerate(sample_ids)}
    population_ids = attack.shadow_manager.sample_random_population_indices(
        num_data=configs.attacker.attack.random_pop_size, exclude=attack.population_exclusions())
    population = attack.shadow_manager.get_random_population(indices=population_ids, labels='original')
    population_probs = attack.compute_p_x_theta(audit_dataset=population,
                                                models_for_sample=[model_ids for _ in population_ids], device=device)

    aucs = []
    for column, (members, non_members) in enumerate(rotations):
        ids = members + non_members
        aucs.append(tuning_aucs(probs[[row[sample] for sample in ids]], ids,
                                [True] * len(members) + [False] * len(non_members), population_probs,
                                population_ids, maps, column, configs.attacker.attack.gamma))
    mean_aucs = np.mean(aucs, axis=0)
    # Ties go to the smallest a, as in the authors' code.
    best = TUNING_A_GRID[int(np.argmax(mean_aucs))]
    print(f'\nRobustMIA a tuning on {settings.dataset}, {len(model_ids)} shadow models, gamma '
          f'{configs.attacker.attack.gamma}: AUC per attacked shadow model, and their mean')
    for index, a in enumerate(TUNING_A_GRID):
        print(f'  a = {a:.1f}: ' + '  '.join(f'{model_aucs[index]:.4f}' for model_aucs in aucs)
              + f'  | mean {mean_aucs[index]:.4f}')
    print(f'Best a for {settings.dataset}: {best}')
    result = {'dataset': settings.dataset, 'best_a': best, 'a_grid': TUNING_A_GRID,
              'aucs': [list(map(float, model_aucs)) for model_aucs in aucs], 'mean_aucs': list(map(float, mean_aucs))}
    path = Path(settings.out_folder) / 'attackers' / get_hash_from_settings(settings, mode='attacker') / 'robust_a.json'
    path.write_text(json.dumps(result, indent=1))
    print(f'Saved to {path}')
    return result


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run every current attack/defense, reusing targets and cleaning temporary artifacts.

Examples (from the repository root):
  python tests/smoke_test_all.py --device cpu --only defense:relax_loss
  python tests/smoke_test_all.py --list-options relax_loss
  python tests/smoke_test_all.py --preset quick --only defense:relax_loss --profiles small_a \\
      --grid defender_optimizer=sgd,sam,esam defender_relax_loss_flatten='*' -- --defender_epochs 2

Anything after `--` is passed to every case (run.py / train_defender.py options, overriding the
profile settings), and --grid runs one case per combination of option values. All the generated
commands are checked against the options of run.py / train_defender.py before anything runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
# Workers import run.py / train_defender.py and the src package from the repository root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoke_edge_cases import EDGE_NAMES, edge_spec

# Distinct algorithms/modes from the current factories and config generator.
# Aliases that produce the same class/config are recorded separately below.
ATTACKS = (
    "online_robust", "offline_robust", "lira", "quantile",
    "neural_feat", "neural_prob", "neural_logit",
    "rmia_loss", "rmia_confidence", "rmia_entropy",
    "pmia_loss", "pmia_confidence", "pmia_entropy",
    "sba_hopskipjump", "sba_qeba-spatial", "sba_qeba-dct",
    "sba_qeba-pca", "sba_qeba-custom",
    "uba_hopskipjump", "uba_qeba-spatial", "uba_qeba-dct",
    "uba_qeba-pca", "uba_qeba-custom", "noise_robustness",
    "transfer_loss", "transfer_confidence", "transfer_entropy",
    "oslo_difgsm", "oslo_mifgsm", "oslo_tifgsm", "oslo_tmifgsm",
    "dh_white", "dh_black", "dh_random", "online_yoqo", "offline_yoqo",
)

ATTACK_ALIASES = {
    "on_robust": "online_robust", "off_robust": "offline_robust",
    "sba": "sba_hopskipjump", "sba_hsj": "sba_hopskipjump",
    "sba_hopskip": "sba_hopskipjump", "sba_hop": "sba_hopskipjump",
    "sba_qeba": "sba_qeba-spatial",
    "uba": "uba_hopskipjump", "uba_hsj": "uba_hopskipjump",
    "uba_hopskip": "uba_hopskipjump", "uba_hop": "uba_hopskipjump",
    "uba_qeba": "uba_qeba-spatial",
    "noise_robust": "noise_robustness", "noise_rob": "noise_robustness",
    "nr": "noise_robustness", "oslo": "oslo_difgsm",
    "dh": "dh_white", "dh-attack": "dh_white", "dh-attack_white": "dh_white",
    "dh-attack_black": "dh_black", "dh-attack_random": "dh_random",
    "on_yoqo": "online_yoqo", "off_yoqo": "offline_yoqo",
    "yoqo": "offline_yoqo",
}

# All distinct current configurations. `none` is the vanilla baseline.
DEFENSES = (
    "none", "dp", "mem_guard", "relax_loss", "adv_reg", "mixup",
    "hamp_train", "hamp_test", "hamp_full", "selena", "mist",
    "mist_mixup", "weighted_smoothing", "purifier", "mmd", "mmd_mixup",
    "ldl", "data_augmentation",
)

DEFENSE_ALIASES = {
    "no": "none", "vanilla": "none",
    "differential_privacy": "dp", "differential-privacy": "dp",
    "memguard": "mem_guard", "mem-guard": "mem_guard",
    "relaxloss": "relax_loss", "relax-loss": "relax_loss",
    "advreg": "adv_reg", "adv-reg": "adv_reg", "hamp": "hamp_full",
    "mist-mixup": "mist_mixup", "weighted-smoothing": "weighted_smoothing",
    "weighted_smooth": "weighted_smoothing", "weighted-smooth": "weighted_smoothing",
    "weightedsmoothing": "weighted_smoothing", "weightedsmooth": "weighted_smoothing",
    "ws": "weighted_smoothing", "mmd-mixup": "mmd_mixup",
    "augmentation": "data_augmentation", "data-augmentation": "data_augmentation",
    "aug": "data_augmentation", "dataaug": "data_augmentation",
    "data_aug": "data_augmentation", "augment": "data_augmentation",
}


@dataclass(frozen=True)
class Profile:
    name: str
    seed: int
    data_limit: int
    batch_size: int
    learning_rate: float
    audit_samples: int
    alpha: float


PROFILES = (
    Profile("small_a", 101, 400, 8, 0.010, 8, 0.20),
    Profile("small_b", 202, 480, 12, 0.005, 10, 0.35),
    Profile("small_c", 303, 560, 16, 0.020, 12, 0.50),
)

# One seed is sufficient for an execution check; the larger presets cover variation.
SMOKE_PROFILES = (Profile("smoke", 101, 400, 16, 0.010, 8, 0.20),)
EDGE_PROFILES = tuple(Profile(name, 101, 400, 16, .010, 8, .20) for name in EDGE_NAMES)

MEDIUM_PROFILES = (
    Profile("medium_a", 101, 1600, 32, 0.010, 64, 0.20),
    Profile("medium_b", 202, 2000, 48, 0.005, 96, 0.35),
    Profile("medium_c", 303, 2400, 64, 0.020, 128, 0.50),
)

# Backward-compatible name for callers that imported the first version directly.
MODERATE_PROFILES = MEDIUM_PROFILES


# Options that the harness sets for every case itself.
RESERVED_OPTIONS = {
    "--defender_mode": "use --only to choose the defenses",
    "--attacker_mode": "use --only to choose the attacks",
    "--out_folder": "every case gets its own output folder",
}


@dataclass(frozen=True)
class Variant:
    """One combination of --grid values, e.g. label 'defender_optimizer=sam'."""
    label: str = ""
    args: tuple[str, ...] = ()

    @property
    def folder(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.label.replace("=", "-").replace(", ", "__"))


@dataclass
class Case:
    kind: str
    component: str
    profile: str
    command: list[str]
    log_file: str
    variant: str = ""
    description: str = ""


@dataclass
class Result:
    kind: str
    component: str
    profile: str
    status: str
    seconds: float
    return_code: int | None
    error: str | None
    log_file: str
    command: list[str]
    variant: str = ""
    description: str = ""


def case_name(case: Case | Result) -> str:
    return f"{case.kind}:{case.component}" + (f" [{case.variant}]" if case.variant else "")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    argv = list(argv)
    extra_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra_args = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(
        description="Smoke-test every current MIA attack and defense without stopping on failures.",
        usage="%(prog)s [options] [-- run.py/train_defender.py options for every case]",
    )
    parser.add_argument("--device", default="0",
                        help="CUDA index (default: 0); pass cpu to disable GPU use")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--datasets-folder", type=Path, default=REPO_ROOT / "datas")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "smoke_test_results")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Maximum seconds for one combination")
    parser.add_argument(
        "--preset", choices=("smoke", "edge", "quick", "medium", "moderate"), default="smoke",
        help=("smoke (default) uses five epochs and one tiny profile; "
              "edge runs five targeted cases per component at five epochs; "
              "quick uses the original one-epoch settings; medium uses ten epochs "
              "and moderately sized stress-test workloads; moderate is an alias for medium"),
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all target, shadow and auxiliary training")
    parser.add_argument("--no-reuse-defender", dest="reuse_defender", action="store_false",
                        help="Train an isolated baseline target for every attack instead of sharing it")
    parser.add_argument("--only", nargs="*", default=None, metavar="KIND:NAME",
                        help="Subset, e.g. attack:lira defense:dp")
    parser.add_argument("--profiles", nargs="+", default=None, metavar="NAME",
                        help="Subset of the profiles of the preset, e.g. small_a")
    parser.add_argument(
        "--grid", nargs="+", action="extend", default=[], metavar="OPTION=V1,V2",
        help=("Run every combination of these run.py/train_defender.py option values, e.g. "
              "defender_optimizer=sgd,sam defender_relax_loss_flatten='*'. '*' takes all the "
              "choices of the option, true/false switch flags on/off, quote values with spaces"),
    )
    parser.add_argument("--list-options", nargs="*", default=None, metavar="PATTERN",
                        help=("List the run.py/train_defender.py options whose name contains a "
                              "pattern (e.g. relax_loss, attacker_oslo), or all of them, and exit"))
    parser.add_argument("--list-components", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--always-zero", action="store_true")
    parser.add_argument(
        "--keep-work", action="store_true",
        help=("Keep checkpoints, shadow data, results, and internal logs. By default "
              "only shared targets survive a case, until their last consumer finishes."),
    )
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    args.extra_args = extra_args
    return args


def project_parser() -> argparse.ArgumentParser:
    """Parser of run.py / train_defender.py, raising ValueError instead of exiting."""
    from src.utils.settings import build_parser

    parser = build_parser()

    def raise_error(message: str):
        raise ValueError(message)

    parser.error = raise_error
    return parser


def option_action(parser: argparse.ArgumentParser, option: str) -> argparse.Action:
    action = parser._option_string_actions.get(option)
    if action is None:
        raise ValueError(f"unknown option {option}; see --list-options")
    return action


def check_not_reserved(option: str) -> None:
    if option in RESERVED_OPTIONS:
        raise ValueError(f"{option} cannot be set here: {RESERVED_OPTIONS[option]}")


def expand_grid(specs: Sequence[str], parser: argparse.ArgumentParser) -> list[Variant]:
    """One Variant per combination of the --grid values (a single empty one without --grid)."""
    axes = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--grid expects OPTION=V1,V2,..., got '{spec}'")
        name, raw_values = spec.split("=", 1)
        option = "--" + name.strip().lstrip("-")
        check_not_reserved(option)
        if option in (axis[0][0] for axis in axes):
            raise ValueError(f"{option} appears twice in --grid")
        action = option_action(parser, option)
        if raw_values.strip() == "*":
            if not action.choices:
                raise ValueError(f"{option}='*' needs an option with a fixed set of choices")
            values = [str(choice) for choice in action.choices]
        else:
            values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not values:
            raise ValueError(f"no values given for {option} in --grid")
        axis = []
        for value in values:
            if action.nargs == 0:  # switch flags such as --defender_nesterov
                if value.lower() not in ("true", "false"):
                    raise ValueError(f"{option} is a switch: use {name}=true,false")
                args = (option,) if value.lower() == "true" else ()
            else:
                args = (option, *shlex.split(value))
            axis.append((option, value, args))
        axes.append(axis)
    variants = []
    for combination in itertools.product(*axes):
        label = ", ".join(f"{option[2:]}={value}" for option, value, _ in combination)
        variants.append(Variant(label, tuple(arg for *_, args in combination for arg in args)))
    return variants


def selected_profiles(args: argparse.Namespace) -> tuple[Profile, ...]:
    profiles = (EDGE_PROFILES if args.preset == "edge" else
                SMOKE_PROFILES if args.preset == "smoke" else
                MEDIUM_PROFILES if args.preset in {"medium", "moderate"} else PROFILES)
    if args.profiles:
        unknown = sorted(set(args.profiles) - {profile.name for profile in profiles})
        if unknown:
            raise ValueError(f"unknown profiles {unknown} for the {args.preset} preset; "
                             f"choose among {[profile.name for profile in profiles]}")
        profiles = tuple(profile for profile in profiles if profile.name in args.profiles)
    return profiles


def print_options(patterns: Sequence[str], parser: argparse.ArgumentParser) -> None:
    patterns = [pattern.lstrip("-") for pattern in patterns]
    shown = 0
    for action in parser._actions:
        option = next((name for name in action.option_strings if name.startswith("--")), None)
        if option is None or option == "--help":
            continue
        if patterns and not any(pattern in option for pattern in patterns):
            continue
        details = [f"choices: {', '.join(map(str, action.choices))}"] if action.choices else []
        details.append("switch" if action.nargs == 0 else f"default: {action.default}")
        print(f"{option}  ({'; '.join(details)})")
        if action.help:
            print(f"    {action.help}")
        shown += 1
    print(f"\n{shown} option(s). Pass them after '--' to set them for every case, or use --grid to sweep them.")


def validate_commands(cases: Sequence[Case], parser: argparse.ArgumentParser) -> list[str]:
    """Errors of the run.py/train_defender.py parser on the commands, one per component and variant."""
    errors = {}
    for case in cases:
        run_args = case.command[case.command.index("--") + 1:]
        try:
            parser.parse_args(run_args)
        except ValueError as exc:
            errors.setdefault(case_name(case), str(exc))
    return [f"{name}: {error}" for name, error in errors.items()]


def common_run_args(profile: Profile, args: argparse.Namespace, out_dir: Path) -> list[str]:
    """Build CLI flags for the selected smoke/stress preset."""
    medium = args.preset in {"medium", "moderate"}
    epochs = str(args.epochs or (5 if args.preset in {"smoke", "edge"} else 10 if medium else 1))
    hidden_layers = ["64", "32"] if medium else ["8"]
    queries = "50" if medium else "3"
    quantiles = "50" if medium else "3"
    auxiliary_batch_size = str(profile.batch_size) if medium else "4"
    robust_alpha = "0.5" if medium else str(profile.alpha)
    statistical_alpha = "0.05" if medium else str(profile.alpha)
    return [
        "--dataset", args.dataset,
        "--datasets_folder", str(args.datasets_folder.resolve()),
        "--out_folder", str(out_dir.resolve()),
        "--device", args.device,
        "--defender_model", "resnet18", "--attacker_model", "resnet18",
        "--defender_epochs", epochs, "--attacker_epochs", epochs,
        "--defender_batch_size", str(profile.batch_size),
        "--attacker_batch_size", str(profile.batch_size),
        "--defender_lr", str(profile.learning_rate),
        "--attacker_lr", str(profile.learning_rate),
        "--defender_lr_sched", "cosine", "--attacker_lr_sched", "cosine",
        "--seed", str(profile.seed),
        "--n_auditing_samples", str(profile.audit_samples), "--audit_in_perc", "0.5",
        "--attacker_robust_rand_pop_size", str(profile.audit_samples),
        "--attacker_robust_alphas", robust_alpha,
        "--attacker_quantile_n", quantiles,
        "--attacker_quantile_low", "0.01" if medium else "0.10",
        "--attacker_quantile_high", "0.99" if medium else "0.90",
        "--attacker_quantile_alpha", statistical_alpha,
        "--attacker_neural_model_layers", *hidden_layers,
        "--attacker_neural_model_epochs", epochs,
        "--attacker_neural_model_lr", str(profile.learning_rate),
        "--attacker_rmia_alpha", statistical_alpha, "--attacker_pmia_alpha", statistical_alpha,
        "--attacker_boundary_n_queries", "256" if args.preset in {"smoke", "edge"} else queries,
        "--attacker_boundary_calibration_samples", "8" if not medium else "32",
        "--attacker_boundary_qeba_reduction_factor", "8" if medium else "4",
        "--attacker_noise_robust_n_queries", queries, "--attacker_noise_robust_sigmas",
        *(["0.01", "0.05", "0.10"] if medium else ["0.05"]),
        "--attacker_oslo_n_models", "4" if medium else "2", "--attacker_oslo_same_arch",
        "--attacker_oslo_source_models_ratio", "0.75" if medium else "0.5",
        "--attacker_oslo_K", "3" if medium else "1",
        "--attacker_oslo_N", "10" if medium else "1",
        "--attacker_dh_n_models", "4" if medium else "2",
        "--attacker_dh_n_queries", queries,
        "--attacker_yoqo_alpha", "2" if medium else str(profile.alpha),
        "--attacker_yoqo_gamma", "5" if medium else "1",
        "--attacker_yoqo_adv_opt_max_iter", "10" if medium else "1",
        "--attacker_yoqo_adv_opt_lr", str(profile.learning_rate),
        "--defender_mem_guard_shadow_model_layers", *hidden_layers,
        "--defender_mem_guard_shadow_model_epochs", epochs,
        "--defender_mem_guard_shadow_model_lr", str(profile.learning_rate),
        "--defender_adv_reg_shadow_attacker_model_layers", *hidden_layers,
        "--defender_adv_reg_shadow_attacker_k", "2" if medium else "1",
        "--defender_selena_K", "5" if medium else "2",
        "--defender_selena_L", "2" if medium else "1",
        "--defender_mist_num_submodels", "4" if medium else "2",
        "--defender_mist_submodel_epochs", "1" if args.preset == "edge" else epochs,
        "--defender_weighted_smoothing_warmup_epochs", "2" if medium else "0",
        "--defender_purifier_reformer_latent_dim", "16" if medium else "4",
        "--defender_purifier_reformer_hidden_dim", "64" if medium else "8",
        "--defender_purifier_reformer_epochs", epochs,
        "--defender_purifier_reformer_lr", str(profile.learning_rate),
        "--defender_purifier_reformer_batch_size", auxiliary_batch_size,
        "--defender_ldl_n_queries", queries,
        "--resume",
    ]


def shadow_count(attack: str) -> int:
    if attack in {"quantile", "pmia_loss", "pmia_confidence", "pmia_entropy",
                  "noise_robustness"}:
        return 1
    if attack.startswith(("sba_", "uba_", "transfer_", "oslo_", "dh_")):
        return 1
    if attack in {"online_robust", "lira", "online_yoqo"} or attack.startswith("neural_"):
        return 4
    return 2


def canonical_filter_name(kind: str, component: str) -> str:
    aliases = ATTACK_ALIASES if kind == "attack" else DEFENSE_ALIASES
    return aliases.get(component, component)


def selected(kind: str, component: str, filters: list[str] | None) -> bool:
    if not filters:
        return True
    for raw in filters:
        if ":" in raw:
            raw_kind, raw_name = raw.split(":", 1)
            raw_kind = "defense" if raw_kind == "defence" else raw_kind
            if raw_kind == kind and canonical_filter_name(kind, raw_name) == component:
                return True
        elif canonical_filter_name(kind, raw) == component:
            return True
    return False


def make_worker_command(profile: Profile, entrypoint: str, run_args: list[str]) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--worker",
            "--data-limit", str(profile.data_limit), "--entrypoint", entrypoint,
            "--", *run_args]


def target_cache_key(run_args: Sequence[str], profile: Profile,
                     parser: argparse.ArgumentParser) -> str:
    """Conservatively identify targets by their final effective training/data settings."""
    settings = vars(parser.parse_args(run_args))
    shared = {key: value for key, value in settings.items()
              if key.startswith("defender_") or key in {
                  "dataset", "datasets_folder", "seed", "data_augmentation", "device",
                  "distributed", "perf_metrics", "perf_metric_to_track"}}
    shared["data_limit"] = profile.data_limit
    return hashlib.sha256(json.dumps(shared, sort_keys=True, default=str).encode()).hexdigest()


def component_args(kind: str, component: str) -> list[str]:
    if kind == "attack":
        return ["--defender_mode", "none", "--attacker_mode", component,
                "--n_shadows", str(shadow_count(component))]
    return ["--defender_mode", component, "--attacker_mode", "quantile", "--n_shadows", "1"]


def build_cases(args: argparse.Namespace, run_root: Path, variants: Sequence[Variant],
                profiles: Sequence[Profile], parser: argparse.ArgumentParser | None = None) -> list[Case]:
    cases: list[Case] = []
    if args.preset == "edge" and parser is None:
        parser = project_parser()
    # Finish one seed/grid combination before starting another, bounding live storage.
    # Run vanilla first, then reuse its completed checkpoints for every attack.
    components = [("defense", "none"), *[("attack", a) for a in ATTACKS],
                  *[("defense", d) for d in DEFENSES if d != "none"]]
    for variant in variants:
        groups: dict[str, list[Case]] = {}
        for profile in profiles:
            for kind, component in components:
                if not selected(kind, component, args.only):
                    continue
                case_path = Path(f"{kind}s", component, variant.folder, profile.name)
                shared_target = args.reuse_defender and (kind == "attack" or component == "none")
                if shared_target and args.preset != "edge":
                    out = run_root / "work" / "baseline" / variant.folder / profile.name
                else:
                    out = run_root / "work" / case_path
                description, edge_args = "", ()
                if args.preset == "edge":
                    edge = edge_spec(kind, component, profile.name, args.epochs or 5)
                    description, edge_args = edge.description, edge.args
                # Later options win: grid > extra options > component edge case > preset.
                run_args = (common_run_args(profile, args, out) + component_args(kind, component)
                            + list(edge_args) + args.extra_args + list(variant.args))
                if shared_target and args.preset == "edge":
                    key = target_cache_key(run_args, profile, parser)
                    out = run_root / "work" / "baseline" / variant.folder / key
                    run_args[run_args.index("--out_folder") + 1] = str(out.resolve())
                entrypoint = "run" if kind == "attack" else "probe_defender"
                command = make_worker_command(profile, entrypoint, run_args)
                if args.preset == "edge":
                    command.insert(command.index("--"), "--strict-checks")
                case = Case(kind, component, profile.name, command,
                            str(run_root / "logs" / f"{case_path}.log"), variant.label, description)
                groups.setdefault(str(out), []).append(case)
        # Execute all consumers of a target together, including consumers in other
        # edge profiles, then remove that target before the next group starts.
        for group in groups.values():
            cases.extend(group)
    return cases


def error_from_log(path: Path, fallback: str) -> str:
    try:
        lines = [x.strip() for x in path.read_text(errors="replace").splitlines() if x.strip()]
    except OSError:
        return fallback
    if not lines:
        return fallback
    traceback_at = next((i for i in range(len(lines) - 1, -1, -1)
                         if lines[i].startswith("Traceback")), None)
    useful = lines[traceback_at:] if traceback_at is not None else lines[-12:]
    return "\n".join(useful[-20:])


def command_option(command: list[str], option: str) -> str | None:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def dump_internal_logs(case: Case, started_wall: float, max_logs: int = 8,
                       tail_lines: int = 60) -> None:
    out_folder = command_option(case.command, "--out_folder")
    if out_folder is None:
        return
    try:
        candidates = [p for p in Path(out_folder).rglob("*.log")
                      if p.is_file() and p.stat().st_mtime >= started_wall - 2]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError as exc:
        print(f"[diagnostics] Could not inspect internal logs: {exc}", flush=True)
        return
    if not candidates:
        print("[diagnostics] No internal project logs were written for this case.", flush=True)
        return
    print("[diagnostics] Tail of internal project log(s):", flush=True)
    for path in candidates[:max_logs]:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            print(f"--- {path} (unreadable: {exc}) ---", flush=True)
            continue
        print(f"--- {path} (last {min(len(lines), tail_lines)} lines) ---", flush=True)
        for line in lines[-tail_lines:]:
            print(line, flush=True)
        print(f"--- end {path} ---", flush=True)


def failure_reason(return_code: int, log_path: Path) -> str:
    if return_code >= 0:
        return error_from_log(log_path, f"process exited {return_code}")
    number = -return_code
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = "UNKNOWN"
    reason = (f"process was terminated by signal {number} ({name}); "
              "a forcibly killed process cannot produce a Python traceback")
    if number == signal.SIGKILL:
        reason += "; common causes are OOM/resource limits or scheduler cancellation"
    return reason


def _run_case(case: Case, timeout: int, index: int, total: int) -> Result:
    log_path = Path(case.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_wall = time.time()
    print(f"[{index}/{total}] {case_name(case)} / {case.profile}", flush=True)
    if case.description:
        print(f"  CHECK: {case.description}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + shlex.join(case.command) + "\n\n")
            if case.description:
                log.write("CHECK: " + case.description + "\n\n")
            log.flush()
            process = subprocess.Popen(
                case.command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                text=True, encoding="utf-8", errors="replace", bufsize=1)

            def tee_output() -> None:
                assert process.stdout is not None
                with process.stdout:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        sys.stdout.write(line)
                        sys.stdout.flush()

            thread = threading.Thread(target=tee_output, daemon=True)
            thread.start()
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                stop_worker(process, signal.SIGKILL)
                process.wait()
                thread.join(timeout=5)
                raise
            except KeyboardInterrupt:
                # Do not leave a worker writing into a directory that the outer
                # per-case cleanup is about to remove.
                stop_worker(process, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stop_worker(process, signal.SIGKILL)
                    process.wait()
                # The worker may exit before a child that ignored SIGTERM.
                stop_worker(process, signal.SIGKILL)
                thread.join(timeout=5)
                raise
            thread.join()
        elapsed = time.monotonic() - started
        if return_code == 0:
            print(f"  PASS ({elapsed:.1f}s)", flush=True)
            return Result(case.kind, case.component, case.profile, "passed", elapsed,
                          return_code, None, case.log_file, case.command, case.variant, case.description)
        reason = failure_reason(return_code, log_path)
        print(f"  ERROR: {reason}", flush=True)
        dump_internal_logs(case, started_wall)
        print(f"  FAIL (exit {return_code}, {elapsed:.1f}s): {reason}", flush=True)
        return Result(case.kind, case.component, case.profile, "failed", elapsed,
                      return_code, reason, case.log_file, case.command, case.variant, case.description)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        reason = f"timed out after {timeout} seconds"
        print(f"  ERROR: {reason}", flush=True)
        dump_internal_logs(case, started_wall)
        print(f"  FAIL ({reason})", flush=True)
        return Result(case.kind, case.component, case.profile, "failed", elapsed,
                      None, reason, case.log_file, case.command, case.variant, case.description)
    except Exception as exc:
        elapsed = time.monotonic() - started
        reason = f"runner error: {type(exc).__name__}: {exc}"
        print(f"  FAIL ({reason})", flush=True)
        return Result(case.kind, case.component, case.profile, "failed", elapsed,
                      None, reason, case.log_file, case.command, case.variant, case.description)


def stop_worker(process: subprocess.Popen, sig: int) -> None:
    """Stop the worker and any children before deleting files they may be writing."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, sig)
        else:
            process.send_signal(sig)
    except ProcessLookupError:
        pass


def cleanup_case_work(case: Case, work_root: Path, keep_defenders: bool = False) -> None:
    """Delete owned artifacts; optionally retain targets with pending consumers."""
    raw_out_folder = command_option(case.command, "--out_folder")
    if raw_out_folder is None:
        print("  CLEANUP WARNING: case has no --out_folder", flush=True)
        return

    out_folder = Path(raw_out_folder).resolve()
    work_root = work_root.resolve()
    try:
        out_folder.relative_to(work_root)
    except ValueError:
        print(f"  CLEANUP WARNING: refusing to delete path outside {work_root}: "
              f"{out_folder}", flush=True)
        return
    if out_folder == work_root:
        print(f"  CLEANUP WARNING: refusing to delete the work root: {work_root}",
              flush=True)
        return

    try:
        if keep_defenders:
            for child in out_folder.iterdir():
                if child.name == "defenders":
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            print("  CLEANED case artifacts; retained shared target checkpoints", flush=True)
        else:
            shutil.rmtree(out_folder)
            print(f"  CLEANED temporary work: {out_folder}", flush=True)
    except FileNotFoundError:
        # A case can fail before creating its output directory.
        pass
    except OSError as exc:
        print(f"  CLEANUP WARNING: could not remove {out_folder}: {exc}", flush=True)


def run_case(case: Case, timeout: int, index: int, total: int,
             work_root: Path, keep_work: bool = False,
             keep_defenders: bool = False) -> Result:
    try:
        return _run_case(case, timeout, index, total)
    finally:
        if not keep_work:
            cleanup_case_work(case, work_root, keep_defenders)


def aggregate(results: list[Result], kind: str,
              profiles_per_component: int) -> tuple[list[str], dict[str, list[Result]]]:
    grouped: dict[str, list[Result]] = {}
    for result in results:
        if result.kind == kind:
            name = result.component + (f" [{result.variant}]" if result.variant else "")
            grouped.setdefault(name, []).append(result)
    passed = sorted(name for name, items in grouped.items()
                    if len(items) == profiles_per_component
                    and all(x.status == "passed" for x in items))
    return passed, {name: items for name, items in sorted(grouped.items())
                    if any(x.status != "passed" for x in items)}


def print_summary(results: list[Result], report_path: Path,
                  profiles_per_component: int) -> bool:
    print("\n" + "=" * 72 + "\nSMOKE TEST SUMMARY\n" + "=" * 72)
    any_failed = False
    for kind in ("attack", "defense"):
        passed, failed = aggregate(results, kind, profiles_per_component)
        print(f"\nPassed {kind}s ({len(passed)}):")
        print("  " + (", ".join(passed) if passed else "none"))
        runs = [result for result in results if result.kind == kind]
        print(f"  Case runs: {sum(r.status == 'passed' for r in runs)} passed, "
              f"{sum(r.status != 'passed' for r in runs)} failed")
        print(f"Failed {kind}s ({len(failed)}):")
        if not failed:
            print("  none")
        for component, items in failed.items():
            any_failed = True
            print(f"  {component}:")
            for item in items:
                if item.status != "passed":
                    reason = (item.error or "unknown error").splitlines()[-1]
                    print(f"    - {item.profile}: {reason} (log: {item.log_file})")
    print(f"\nFull JSON report: {report_path}")
    return any_failed


def write_report(results: list[Result], report_path: Path, started_at: str,
                 preset: str, profiles_per_component: int, options: dict | None = None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "profiles_per_component": profiles_per_component,
        **(options or {}),
        "attack_matrix": list(ATTACKS), "defense_matrix": list(DEFENSES),
        "attack_aliases_not_repeated": ATTACK_ALIASES,
        "defense_aliases_not_repeated": DEFENSE_ALIASES,
        "results": [asdict(result) for result in results],
    }
    for kind in ("attack", "defense"):
        passed, failed = aggregate(results, kind, profiles_per_component)
        payload[f"passed_{kind}s"] = passed
        payload[f"failed_{kind}s"] = {
            name: [asdict(item) for item in items if item.status != "passed"]
            for name, items in failed.items()
        }
    payload["counts"] = {
        "passed": sum(r.status == "passed" for r in results),
        "failed": sum(r.status != "passed" for r in results),
        "completed": len(results),
        "not_run": payload.get("planned_cases", len(results)) - len(results),
        **{f"{status}_{kind}s": len(payload[f"{status}_{kind}s"])
           for kind in ("attack", "defense") for status in ("passed", "failed")},
        **{f"{status}_{kind}_cases": sum(r.kind == kind and r.status == status for r in results)
           for kind in ("attack", "defense") for status in ("passed", "failed")},
    }
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)


def validate_attack_outputs(scores, decisions, n_samples: int) -> None:
    """Check the numerical contract before a metric routine can cast/coerce outputs."""
    import numpy as np

    scores = np.asarray(scores).reshape(-1)
    if len(scores) != n_samples:
        raise ValueError(f"Expected {n_samples} membership scores, got shape {scores.shape}")
    invalid = np.flatnonzero(~np.isfinite(scores))
    if len(invalid):
        raise ValueError(f"Non-finite membership scores: {len(invalid)}/{n_samples}, "
                         f"audit positions {invalid[:8].tolist()}")
    if decisions is not None:
        decisions = np.asarray(decisions).reshape(-1)
        if len(decisions) != n_samples or not np.isin(decisions, [0, 1]).all():
            raise ValueError("Membership decisions must match audit length and contain only 0 or 1")


def install_strict_attack_checks():
    """Instrument only this worker; leave production attack behavior unchanged."""
    import numpy as np
    from src.mia.attacks.base_mia import BaseMIA

    original_stats = BaseMIA.compute_stats
    calls = []

    def checked_stats(attack, scores, decisions=None, params=None):
        n_samples = len(attack.audit_manager.get_all_ids())
        validate_attack_outputs(scores, decisions, n_samples)
        metrics = original_stats(attack, scores, decisions=decisions, params=params)
        if not np.isfinite(metrics["auc"]) or not 0 <= metrics["auc"] <= 1:
            raise ValueError("Attack evaluation must produce a finite AUC in [0, 1]")
        counts = (metrics[k] for k in ("true_positives", "true_negatives", "false_positives", "false_negatives"))
        if sum(counts) != n_samples:
            raise ValueError("Attack confusion counts do not cover the audit set")
        calls.append(n_samples)
        print(f"[edge check] validated {n_samples} scores, decisions and metrics", flush=True)
        return metrics

    BaseMIA.compute_stats = checked_stats
    return calls


def probe_defender(strict_checks: bool = False) -> None:
    """Train/apply a defense and execute member/nonmember queries through its wrapper."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from src.utils.settings import gather_settings, setup_configs_and_folder_from_settings
    from src.mia.defenses import get_defender_class

    settings = gather_settings()
    configs, _ = setup_configs_and_folder_from_settings(settings)
    defender = get_defender_class(settings.defender_mode)(defender_configs=configs.defender)
    defender.train_model(train_configs=configs.defender.train, return_stats=True)
    defender.defend_model(device=configs.defender.train.device)
    device = defender.get_device(settings.device)
    model = defender.get_defended_model().to(device).eval()
    if strict_checks and any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Defended model has non-finite parameters after training")
    for split in ("train", "test"):
        dataset = defender.get_dataset().get(split)
        subset = Subset(dataset, range(min(4, len(dataset))))
        if not len(subset):
            raise ValueError(f"Defense probe has an empty {split} split")
        inputs, _, _, _ = next(iter(DataLoader(subset, batch_size=len(subset))))
        # MemGuard needs autograd inside inference; do not use inference_mode here.
        inputs = inputs.to(device)
        probes = [("batch", inputs)]
        if strict_checks:
            probes += [("singleton", inputs[:1]), ("odd_batch", inputs[:3]),
                       ("permuted", inputs.flip(0)), ("repeat", inputs),
                       ("zeros", torch.zeros_like(inputs[:1])), ("ones", torch.ones_like(inputs[:1]))]
        for label, batch in probes:
            original = batch.clone()
            outputs = model(batch)
            expected = (len(batch), configs.defender.model.num_classes)
            if tuple(outputs.shape) != expected or not torch.isfinite(outputs).all():
                raise ValueError(f"Invalid defended outputs on {split}/{label}: expected finite {expected}, "
                                 f"got {tuple(outputs.shape)}")
            if strict_checks and not torch.equal(batch, original):
                raise ValueError(f"Defended model mutated its input on {split}/{label}")
        print(f"[smoke worker] defense {settings.defender_mode}: {split} inference PASS", flush=True)


def worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--data-limit", type=int, required=True)
    parser.add_argument("--entrypoint", choices=("run", "train_defender", "probe_defender"), required=True)
    parser.add_argument("--strict-checks", action="store_true")
    worker_args, run_args = parser.parse_known_args(argv)
    if run_args and run_args[0] == "--":
        run_args = run_args[1:]

    # Patch the current central data importer in the isolated child only.
    import src.data as data_module
    import src.mia.helpers.shadow_data_manager as shadow_data_module
    from src.data.helpers import SubsampledDataset

    try:
        out_index = run_args.index("--out_folder") + 1
        smoke_shadow_dir = Path(run_args[out_index]) / "shadow_datas"
    except (ValueError, IndexError):
        smoke_shadow_dir = REPO_ROOT / "smoke_test_results" / "worker_shadow_datas"
    shadow_data_module.DEFAULT_SHADOW_DATASETS_FOLDER = smoke_shadow_dir

    # Label-only threshold training requests equally many shadow nonmembers.
    # Production's 0.6 fraction makes that impossible; 0.4 is smoke-only sizing.
    shadow_data_module.PERCENTAGE_OF_DATA_TO_USE_FOR_SHADOW_DATASETS = 0.4
    original_import = data_module.import_dataset_by_name

    def import_limited_dataset(*args, **kwargs):
        dataset = original_import(*args, **kwargs)
        if worker_args.data_limit > len(dataset):
            raise ValueError(f"smoke limit {worker_args.data_limit} exceeds dataset size {len(dataset)}")
        ids = dataset.get_indices(mode="original")[:worker_args.data_limit]
        limited = SubsampledDataset(dataset, original_indices=ids, strict=True)
        print(f"[smoke worker] limited merged dataset to {len(limited)} samples", flush=True)
        return limited

    data_module.import_dataset_by_name = import_limited_dataset
    if worker_args.strict_checks:
        # BaseDefender builds its model before TrainManager seeds the RNG. Seed
        # here too, so an edge failure can be reproduced from the recorded CLI.
        import random
        import numpy as np
        import torch

        seed = project_parser().parse_args(run_args).seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"[edge check] seeded model initialization with {seed}", flush=True)
    sys.argv = [str(REPO_ROOT / f"{worker_args.entrypoint}.py"), *run_args]
    try:
        if worker_args.entrypoint == "probe_defender":
            probe_defender(strict_checks=worker_args.strict_checks)
            return 0
        elif worker_args.entrypoint == "run":
            import run as entrypoint_module
        else:
            import train_defender as entrypoint_module
        checks = install_strict_attack_checks() if worker_args.strict_checks else None
        entrypoint_module.main()
        if checks is not None and not checks:
            raise ValueError("Attack completed without evaluating any membership scores")
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


def print_component_list() -> None:
    print(f"Attacks ({len(ATTACKS)} distinct modes):\n  " + "\n  ".join(ATTACKS))
    print(f"\nDefenses ({len(DEFENSES)} configurations, including vanilla):\n  "
          + "\n  ".join(DEFENSES))
    print(f"\nDuplicate CLI aliases not rerun: {len(ATTACK_ALIASES)} attack aliases, "
          f"{len(DEFENSE_ALIASES)} defense aliases")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return worker_main(argv)
    args = parse_args(argv)
    if args.list_components:
        print_component_list()
        return 0
    try:
        parser = project_parser()
        if args.list_options is not None:
            print_options(args.list_options, parser)
            return 0
        for token in args.extra_args:
            if token.startswith("--"):
                check_not_reserved(token.split("=", 1)[0])
        variants = expand_grid(args.grid, parser)
        profiles = selected_profiles(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    run_root = args.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = run_root / "report.json"
    try:
        cases = build_cases(args, run_root, variants, profiles, parser=parser)
    except ValueError as exc:
        print(f"Invalid case settings; nothing was run: {exc}", file=sys.stderr)
        return 2
    profiles_per_component = len(profiles)
    if not cases:
        print("No cases matched --only. Use --list-components for valid names.", file=sys.stderr)
        return 2
    errors = validate_commands(cases, parser)
    if errors:
        print("Invalid run.py/train_defender.py options (see --list-options); nothing was run:",
              file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 2
    options = {"profiles": [profile.name for profile in profiles], "grid": args.grid,
               "extra_args": args.extra_args, "epochs_override": args.epochs,
               "reuse_defender": args.reuse_defender, "planned_cases": len(cases),
               "scope": "each attack against vanilla; each defense trained/applied/probed separately"}
    options["case_plan"] = [dict(kind=c.kind, component=c.component, profile=c.profile,
                                 variant=c.variant, description=c.description, log_file=c.log_file)
                            for c in cases]
    print(f"Prepared {len(cases)} cases ({len(variants)} option combination(s) x "
          f"{profiles_per_component} profile(s) per component, {args.preset} preset).")
    print(f"Results directory: {run_root}")
    if args.keep_work:
        print("Temporary per-case work will be kept (--keep-work).")
    else:
        print("Temporary attack data/checkpoints are deleted after each case; shared targets "
              "are deleted after their last consumer.")
    if args.reuse_defender:
        print("Compatible baseline targets are reused across attacks and grouped by training settings. "
              "Attack-specific shadows are trained separately.")
    if args.dry_run:
        for case in cases:
            if case.description:
                print(f"  CHECK: {case.description}")
            print(f"{case_name(case)}/{case.profile}: {shlex.join(case.command)}")
        return 0

    started_at = datetime.now(timezone.utc).isoformat()
    results: list[Result] = []
    remaining = Counter(command_option(case.command, "--out_folder") for case in cases)
    interrupted = False

    def on_terminate(_signum, _frame):
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, on_terminate)
    try:
        for index, case in enumerate(cases, start=1):
            out = command_option(case.command, "--out_folder")
            remaining[out] -= 1
            results.append(run_case(case, args.timeout, index, len(cases),
                                    run_root / "work", args.keep_work,
                                    keep_defenders=remaining[out] > 0))
            write_report(results, report_path, started_at, args.preset,
                         profiles_per_component, options)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; writing the partial report.", file=sys.stderr)
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        # Also release shared targets if interrupted between their consumers.
        if not args.keep_work:
            for case in cases:
                cleanup_case_work(case, run_root / "work")
        options["interrupted"] = interrupted
        write_report(results, report_path, started_at, args.preset,
                     profiles_per_component, options)
    any_failed = print_summary(results, report_path, profiles_per_component)
    print(f"\nCases: {sum(r.status == 'passed' for r in results)} passed, "
          f"{sum(r.status != 'passed' for r in results)} failed, "
          f"{len(cases) - len(results)} not run.")
    if interrupted:
        return 130
    return 0 if args.always_zero or not any_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

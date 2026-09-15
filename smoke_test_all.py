#!/usr/bin/env python3
"""Run every current attack/defense with local test profiles, continuing on errors."""

from __future__ import annotations

import argparse
import json
import os
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


REPO_ROOT = Path(__file__).resolve().parent

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

MEDIUM_PROFILES = (
    Profile("medium_a", 101, 1600, 32, 0.010, 64, 0.20),
    Profile("medium_b", 202, 2000, 48, 0.005, 96, 0.35),
    Profile("medium_c", 303, 2400, 64, 0.020, 128, 0.50),
)

# Backward-compatible name for callers that imported the first version directly.
MODERATE_PROFILES = MEDIUM_PROFILES


@dataclass
class Case:
    kind: str
    component: str
    profile: str
    command: list[str]
    log_file: str


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


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test every current MIA attack and defense without stopping on failures."
    )
    parser.add_argument("--device", default="0",
                        help="CUDA index (default: 0); pass cpu to disable GPU use")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--datasets-folder", type=Path, default=REPO_ROOT / "datas")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "smoke_test_results")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Maximum seconds for one combination")
    parser.add_argument(
        "--preset", choices=("quick", "medium", "moderate"), default="quick",
        help=("quick uses the original one-epoch settings; medium uses ten epochs "
              "and moderately sized stress-test workloads; moderate is an alias for medium"),
    )
    parser.add_argument("--only", nargs="*", default=None, metavar="KIND:NAME",
                        help="Subset, e.g. attack:lira defense:dp")
    parser.add_argument("--list-components", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--always-zero", action="store_true")
    parser.add_argument(
        "--keep-work", action="store_true",
        help=("Keep per-case checkpoints, shadow data, results, and internal logs. "
              "By default these temporary artifacts are deleted after each case."),
    )
    return parser.parse_args(argv)


def common_run_args(profile: Profile, args: argparse.Namespace, out_dir: Path) -> list[str]:
    """Build CLI flags for either the quick or medium local stress preset."""
    medium = args.preset in {"medium", "moderate"}
    epochs = "10" if medium else "1"
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
        "--attacker_boundary_n_queries", queries,
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
        "--defender_mist_submodel_epochs", epochs,
        "--defender_weighted_smoothing_warmup_epochs", "2" if medium else "0",
        "--defender_purifier_reformer_latent_dim", "16" if medium else "4",
        "--defender_purifier_reformer_hidden_dim", "64" if medium else "8",
        "--defender_purifier_reformer_epochs", epochs,
        "--defender_purifier_reformer_lr", str(profile.learning_rate),
        "--defender_purifier_reformer_batch_size", auxiliary_batch_size,
        "--defender_purifier_pindex_size", (str(profile.audit_samples) if medium else "8"),
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


def build_cases(args: argparse.Namespace, run_root: Path) -> list[Case]:
    cases: list[Case] = []
    logs = run_root / "logs"
    profiles = MEDIUM_PROFILES if args.preset in {"medium", "moderate"} else PROFILES
    for attack in ATTACKS:
        if not selected("attack", attack, args.only):
            continue
        for profile in profiles:
            # Isolate every case so its checkpoints and shadow data can be deleted
            # immediately without touching another case's files.
            out = run_root / "work" / "attacks" / attack / profile.name
            run_args = common_run_args(profile, args, out)
            run_args += ["--defender_mode", "none", "--attacker_mode", attack,
                         "--n_shadows", str(shadow_count(attack))]
            log_file = logs / "attacks" / attack / f"{profile.name}.log"
            cases.append(Case("attack", attack, profile.name,
                              make_worker_command(profile, "run", run_args), str(log_file)))
    # This entrypoint isolates the defense test from failures in a probe attack.
    for defense in DEFENSES:
        if not selected("defense", defense, args.only):
            continue
        for profile in profiles:
            out = run_root / "work" / "defenses" / defense / profile.name
            run_args = common_run_args(profile, args, out)
            run_args += ["--defender_mode", defense, "--attacker_mode", "quantile",
                         "--n_shadows", "1"]
            log_file = logs / "defenses" / defense / f"{profile.name}.log"
            cases.append(Case("defense", defense, profile.name,
                              make_worker_command(profile, "train_defender", run_args), str(log_file)))
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
    print(f"[{index}/{total}] {case.kind}:{case.component} / {case.profile}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + shlex.join(case.command) + "\n\n")
            log.flush()
            process = subprocess.Popen(
                case.command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                text=True, encoding="utf-8", errors="replace", bufsize=1)

            def tee_output() -> None:
                assert process.stdout is not None
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
                process.kill()
                process.wait()
                thread.join(timeout=5)
                raise
            except KeyboardInterrupt:
                # Do not leave a worker writing into a directory that the outer
                # per-case cleanup is about to remove.
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                thread.join(timeout=5)
                raise
            thread.join()
        elapsed = time.monotonic() - started
        if return_code == 0:
            print(f"  PASS ({elapsed:.1f}s)", flush=True)
            return Result(case.kind, case.component, case.profile, "passed", elapsed,
                          return_code, None, case.log_file, case.command)
        reason = failure_reason(return_code, log_path)
        print(f"  ERROR: {reason}", flush=True)
        dump_internal_logs(case, started_wall)
        print(f"  FAIL (exit {return_code}, {elapsed:.1f}s): {reason}", flush=True)
        return Result(case.kind, case.component, case.profile, "failed", elapsed,
                      return_code, reason, case.log_file, case.command)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        reason = f"timed out after {timeout} seconds"
        print(f"  ERROR: {reason}", flush=True)
        dump_internal_logs(case, started_wall)
        print(f"  FAIL ({reason})", flush=True)
        return Result(case.kind, case.component, case.profile, "failed", elapsed,
                      None, reason, case.log_file, case.command)
    except Exception as exc:
        elapsed = time.monotonic() - started
        reason = f"runner error: {type(exc).__name__}: {exc}"
        print(f"  FAIL ({reason})", flush=True)
        return Result(case.kind, case.component, case.profile, "failed", elapsed,
                      None, reason, case.log_file, case.command)


def cleanup_case_work(case: Case, work_root: Path) -> None:
    """Delete one case's isolated artifacts, refusing paths outside work_root."""
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
        shutil.rmtree(out_folder)
        print(f"  CLEANED temporary work: {out_folder}", flush=True)
    except FileNotFoundError:
        # A case can fail before creating its output directory.
        pass
    except OSError as exc:
        print(f"  CLEANUP WARNING: could not remove {out_folder}: {exc}", flush=True)


def run_case(case: Case, timeout: int, index: int, total: int,
             work_root: Path, keep_work: bool = False) -> Result:
    try:
        return _run_case(case, timeout, index, total)
    finally:
        if not keep_work:
            cleanup_case_work(case, work_root)


def aggregate(results: list[Result], kind: str,
              profiles_per_component: int) -> tuple[list[str], dict[str, list[Result]]]:
    grouped: dict[str, list[Result]] = {}
    for result in results:
        if result.kind == kind:
            grouped.setdefault(result.component, []).append(result)
    passed = sorted(name for name, items in grouped.items()
                    if len(items) == profiles_per_component
                    and all(x.status == "passed" for x in items))
    return passed, {name: items for name, items in sorted(grouped.items()) if name not in passed}


def print_summary(results: list[Result], report_path: Path,
                  profiles_per_component: int) -> bool:
    print("\n" + "=" * 72 + "\nSMOKE TEST SUMMARY\n" + "=" * 72)
    any_failed = False
    for kind in ("attack", "defense"):
        passed, failed = aggregate(results, kind, profiles_per_component)
        print(f"\nPassed {kind}s ({len(passed)}):")
        print("  " + (", ".join(passed) if passed else "none"))
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
                 preset: str, profiles_per_component: int) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "profiles_per_component": profiles_per_component,
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
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--data-limit", type=int, required=True)
    parser.add_argument("--entrypoint", choices=("run", "train_defender"), required=True)
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
    sys.argv = [str(REPO_ROOT / f"{worker_args.entrypoint}.py"), *run_args]
    try:
        if worker_args.entrypoint == "run":
            import run as entrypoint_module
        else:
            import train_defender as entrypoint_module
        entrypoint_module.main()
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
    run_root = args.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = run_root / "report.json"
    cases = build_cases(args, run_root)
    profiles_per_component = len(
        MEDIUM_PROFILES if args.preset in {"medium", "moderate"} else PROFILES
    )
    if not cases:
        print("No cases matched --only. Use --list-components for valid names.", file=sys.stderr)
        return 2
    print(f"Prepared {len(cases)} cases ({profiles_per_component} combinations per component, "
          f"{args.preset} preset).")
    print(f"Results directory: {run_root}")
    if args.keep_work:
        print("Temporary per-case work will be kept (--keep-work).")
    else:
        print("Temporary per-case checkpoints and data will be deleted after each case.")
    if args.dry_run:
        for case in cases:
            print(f"{case.kind}:{case.component}/{case.profile}: {shlex.join(case.command)}")
        return 0

    started_at = datetime.now(timezone.utc).isoformat()
    results: list[Result] = []
    try:
        for index, case in enumerate(cases, start=1):
            results.append(run_case(case, args.timeout, index, len(cases),
                                    run_root / "work", args.keep_work))
            write_report(results, report_path, started_at, args.preset,
                         profiles_per_component)
    except KeyboardInterrupt:
        print("\nInterrupted; writing the partial report.", file=sys.stderr)
    finally:
        write_report(results, report_path, started_at, args.preset,
                     profiles_per_component)
    any_failed = print_summary(results, report_path, profiles_per_component)
    return 0 if args.always_zero or not any_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

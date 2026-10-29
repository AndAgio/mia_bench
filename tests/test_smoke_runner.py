"""Dependency-free regressions for scheduling, model lifetime and failure reporting.

Run: python tests/test_smoke_runner.py
These check the harness, not the PyTorch attack/defense implementations.
"""
import argparse
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import smoke_test_all as smoke


def cli_parser():
    # Exercise the actual project's flag definitions without importing its training
    # stack. settings.py's build_parser uses only argparse, pathlib and constants.
    path = smoke.REPO_ROOT / "src/utils/settings.py"
    definition = next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == "build_parser")
    namespace = runpy.run_path(str(smoke.REPO_ROOT / "src/utils/variables.py"))
    namespace.update(argparse=argparse, pathlib=__import__("pathlib"))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["build_parser"]()


class SmokeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parser = cli_parser()

    def cases(self, flags=()):
        args = smoke.parse_args(flags)
        cases = smoke.build_cases(args, self.root,
                                 smoke.expand_grid(args.grid, self.parser),
                                 smoke.selected_profiles(args), parser=self.parser)
        return args, cases

    def settings(self, case):
        return self.parser.parse_args(case.command[case.command.index("--") + 1:])

    def test_default_covers_factory_modes_and_five_epoch_training(self):
        _, cases = self.cases()
        self.assertEqual(len(cases), 54)
        self.assertEqual({c.component for c in cases if c.kind == "attack"}, set(smoke.ATTACKS))
        self.assertEqual({c.component for c in cases if c.kind == "defense"}, set(smoke.DEFENSES))
        self.assertFalse(smoke.validate_commands(cases, self.parser))
        for case in cases:
            settings = self.settings(case)
            for option in ("defender_epochs", "attacker_epochs", "attacker_neural_model_epochs",
                           "defender_mem_guard_shadow_model_epochs", "defender_mist_submodel_epochs",
                           "defender_purifier_reformer_epochs"):
                self.assertEqual(getattr(settings, option), 5)
            self.assertTrue(settings.resume)
            if case.kind == "defense":
                self.assertIn("probe_defender", case.command)
        # Also detect a newly added factory mode not represented by our coverage list.
        for kind, names, aliases in (("attacks", smoke.ATTACKS, smoke.ATTACK_ALIASES),
                                     ("defenses", smoke.DEFENSES, smoke.DEFENSE_ALIASES)):
            tree = ast.parse((smoke.REPO_ROOT / f"src/mia/{kind}/__init__.py").read_text())
            factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
            modes = {n.value for comparison in ast.walk(factory) if isinstance(comparison, ast.Compare)
                     for n in ast.walk(comparison) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            self.assertEqual(modes, set(names) | set(aliases))

    def test_targets_shared_but_training_settings_and_logs_agree(self):
        _, cases = self.cases()
        baseline, *attacks = cases[:37]
        self.assertEqual((baseline.kind, baseline.component), ("defense", "none"))
        target = self.settings(baseline)
        for case in attacks:
            settings = self.settings(case)
            self.assertEqual(settings.out_folder, target.out_folder)
            defender_options = {k: v for k, v in vars(settings).items() if k.startswith("defender_")}
            self.assertEqual(defender_options,
                             {k: v for k, v in vars(target).items() if k.startswith("defender_")})
        self.assertEqual(len({c.log_file for c in cases}), 54)
        self.assertEqual(len({self.settings(c).out_folder for c in cases}), 18)

    def test_groups_never_share_targets_with_other_profiles_or_grid_values(self):
        _, cases = self.cases(["--preset", "quick", "--profiles", "small_a", "small_b",
                               "--only", "attack:lira", "attack:quantile", "defense:none",
                               "--grid", "defender_lr=0.01,0.02"])
        self.assertEqual(len(cases), 12)
        paths = [self.settings(c).out_folder for c in cases]
        self.assertEqual(len(set(paths)), 4)
        for offset in range(0, 12, 3):
            self.assertEqual(len(set(paths[offset:offset + 3])), 1)
        _, isolated = self.cases(["--no-reuse-defender"])
        self.assertEqual(len({self.settings(c).out_folder for c in isolated}), 54)

    def test_epochs_and_option_precedence(self):
        _, cases = self.cases(["--epochs", "3", "--only", "attack:lira",
                               "--grid", "defender_epochs=2", "--", "--defender_epochs", "4"])
        settings = self.settings(cases[0])
        self.assertEqual(settings.defender_epochs, 2)
        self.assertEqual(settings.attacker_epochs, 3)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                smoke.parse_args(["--epochs", "0"])

    def fake_case(self, component, code, out=None):
        out = out or self.root / "work/shared"
        return smoke.Case("attack", component, "smoke",
                          [sys.executable, "-c", code, "--", "--out_folder", str(out)],
                          str(self.root / f"logs/{component}.log"))

    def test_subprocess_reuse_cleanup_and_failure(self):
        code = """
import pathlib, sys
out = pathlib.Path(sys.argv[-1])
checkpoint = out / 'defenders/target/last.pth'
checkpoint.parent.mkdir(parents=True, exist_ok=True)
if checkpoint.exists():
    print('REUSED TARGET')
else:
    checkpoint.write_text('trained once')
    print('TRAINED TARGET')
for name in ('attackers/shadow.pth', 'shadow_datas/map.pkl', 'experiments/results.json'):
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('temporary')
"""
        results = []
        for i, suffix in enumerate(("", "", "\nraise RuntimeError('attack failed')")):
            case = self.fake_case(f"attack_{i}", code + suffix)
            with redirect_stdout(io.StringIO()):
                results.append(smoke.run_case(case, 10, i + 1, 3, self.root / "work",
                                              keep_defenders=i < 2))
            out = Path(smoke.command_option(case.command, "--out_folder"))
            if i < 2:
                self.assertEqual([p.name for p in out.iterdir()], ["defenders"])
                self.assertEqual((out / "defenders/target/last.pth").read_text(), "trained once")
            else:
                self.assertFalse(out.exists())
            log = Path(case.log_file).read_text()
            self.assertIn("TRAINED TARGET" if i == 0 else "REUSED TARGET", log)
        self.assertEqual([r.status for r in results], ["passed", "passed", "failed"])
        self.assertIn("attack failed", results[-1].error)
        report = self.root / "report.json"
        smoke.write_report(results, report, "now", "smoke", 1, {"planned_cases": 4})
        counts = json.loads(report.read_text())["counts"]
        self.assertEqual((counts["passed"], counts["failed"], counts["not_run"]), (2, 1, 1))

    def test_cleanup_refuses_external_paths_and_does_not_follow_child_symlinks(self):
        outside = self.root / "existing_models"
        outside.mkdir()
        marker = outside / "saved.pth"
        marker.write_text("keep")
        case = self.fake_case("outside", "", outside)
        with redirect_stdout(io.StringIO()):
            smoke.cleanup_case_work(case, self.root / "work")
        self.assertTrue(marker.exists())
        owned = self.root / "work/owned"
        owned.mkdir(parents=True)
        (owned / "attackers").symlink_to(outside, target_is_directory=True)
        case = self.fake_case("owned", "", owned)
        with redirect_stdout(io.StringIO()):
            smoke.cleanup_case_work(case, self.root / "work", keep_defenders=True)
        self.assertTrue(marker.exists())
        self.assertFalse((owned / "attackers").exists())

    def test_timeout_cleans_work_and_keeps_log(self):
        case = self.fake_case("timeout", "import pathlib, sys, time; "
                              "pathlib.Path(sys.argv[-1]).mkdir(parents=True); time.sleep(30)")
        with redirect_stdout(io.StringIO()):
            result = smoke.run_case(case, 1, 1, 1, self.root / "work")
        self.assertEqual(result.status, "failed")
        self.assertIn("timed out", result.error)
        self.assertFalse((self.root / "work/shared").exists())
        self.assertTrue(Path(case.log_file).exists())

    def test_interrupted_run_reports_unrun_cases_and_cleans_shared_target(self):
        def interrupt(case, *args, **kwargs):
            out = Path(smoke.command_option(case.command, "--out_folder"))
            (out / "defenders").mkdir(parents=True)
            raise KeyboardInterrupt

        with patch.object(smoke, "project_parser", return_value=self.parser), \
                patch.object(smoke, "run_case", side_effect=interrupt), \
                redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            status = smoke.main(["--output-dir", str(self.root), "--only", "attack:lira"])
        self.assertEqual(status, 130)
        report, = self.root.glob("*/report.json")
        payload = json.loads(report.read_text())
        self.assertTrue(payload["interrupted"])
        self.assertEqual(payload["counts"]["not_run"], 1)
        self.assertFalse(list(report.parent.glob("work/**/defenders")))

    def test_main_continues_after_failure_and_releases_target_at_last_use(self):
        build_cases = smoke.build_cases

        def fake_workers(args, root, variants, profiles, parser=None):
            cases = build_cases(args, root, variants, profiles, parser=parser)
            for case in cases:
                code = """
import pathlib, sys
out = pathlib.Path(sys.argv[-1])
checkpoint = out / 'defenders/target/last.pth'
checkpoint.parent.mkdir(parents=True, exist_ok=True)
print('REUSED' if checkpoint.exists() else 'TRAINED')
checkpoint.write_text('target')
(out / 'attackers').mkdir(exist_ok=True)
"""
                if case.component == "quantile":
                    code += "\nraise ValueError('expected failure')"
                case.command = [sys.executable, "-c", code, "--", "--out_folder",
                                smoke.command_option(case.command, "--out_folder")]
            return cases

        with patch.object(smoke, "project_parser", return_value=self.parser), \
                patch.object(smoke, "build_cases", side_effect=fake_workers), \
                redirect_stdout(io.StringIO()):
            status = smoke.main(["--output-dir", str(self.root), "--only", "attack:lira",
                                 "attack:quantile", "attack:noise_robustness"])
        self.assertEqual(status, 1)
        report, = self.root.glob("*/report.json")
        payload = json.loads(report.read_text())
        self.assertEqual([r["status"] for r in payload["results"]], ["passed", "failed", "passed"])
        self.assertEqual(payload["counts"]["not_run"], 0)
        self.assertEqual(payload["counts"]["passed_attacks"], 2)
        self.assertEqual(payload["counts"]["failed_attacks"], 1)
        for index, result in enumerate(payload["results"]):
            self.assertIn("TRAINED" if index == 0 else "REUSED", Path(result["log_file"]).read_text())
            self.assertFalse(Path(smoke.command_option(result["command"], "--out_folder")).exists())

    def test_edge_matrix_has_five_distinct_valid_cases_per_component(self):
        from collections import Counter
        _, cases = self.cases(["--preset", "edge"])
        self.assertEqual(len(cases), 270)
        self.assertEqual(set(Counter((c.kind, c.component) for c in cases).values()), {5})
        self.assertFalse(smoke.validate_commands(cases, self.parser))
        for kind, components in (("attack", smoke.ATTACKS), ("defense", smoke.DEFENSES)):
            for component in components:
                subset = [c for c in cases if c.kind == kind and c.component == component]
                self.assertEqual({c.profile for c in subset}, set(smoke.EDGE_NAMES))
                self.assertEqual(len({tuple(c.command) for c in subset}), 5)
                signatures = [json.dumps({k: v for k, v in vars(self.settings(c)).items()
                                          if k != "out_folder"}, sort_keys=True, default=str) for c in subset]
                self.assertEqual(len(set(signatures)), 5)
                self.assertTrue(all(c.description and "--strict-checks" in c.command for c in subset))
        for case in cases:
            settings = self.settings(case)
            self.assertEqual(settings.defender_epochs, 5)
            self.assertEqual(settings.attacker_epochs, 5)
            members = int(settings.n_auditing_samples * settings.audit_in_perc)
            self.assertGreater(members, 0)
            self.assertLess(members, settings.n_auditing_samples)
            if case.kind == "defense" and case.component == "relax_loss":
                self.assertGreater(settings.defender_relax_loss_alpha, 0)

    def test_edge_targets_shared_across_profiles_and_consumers_contiguous(self):
        _, cases = self.cases(["--preset", "edge"])
        attacks = [c for c in cases if c.kind == "attack"]
        target_paths = {self.settings(c).out_folder for c in attacks}
        self.assertEqual(len(target_paths), 1)
        baseline = next(c for c in cases if c.kind == "defense" and c.component == "none"
                        and c.profile == "baseline")
        self.assertIn(self.settings(baseline).out_folder, target_paths)
        visited, previous = set(), None
        for case in cases:
            path = self.settings(case).out_folder
            if path != previous:
                self.assertNotIn(path, visited)
                visited.add(path)
            previous = path
        _, isolated = self.cases(["--preset", "edge", "--no-reuse-defender"])
        self.assertEqual(len({self.settings(c).out_folder for c in isolated}), 270)

    def test_edge_target_key_tracks_effective_seed_data_training_and_limits(self):
        args, cases = self.cases(["--preset", "edge", "--only", "attack:pmia_loss"])
        case = cases[0]
        flags = case.command[case.command.index("--") + 1:]
        profile = smoke.selected_profiles(args)[0]
        key = smoke.target_cache_key(flags, profile, self.parser)
        for override in (("--seed", "999"), ("--defender_batch_size", "7"),
                         ("--defender_data_split_perc", ".6"), ("--datasets_folder", "/tmp/other-data")):
            self.assertNotEqual(key, smoke.target_cache_key(flags + list(override), profile, self.parser))
        for override in (("--n_shadows", "6"), ("--audit_in_perc", ".25"),
                         ("--attacker_optimizer", "adam")):
            self.assertEqual(key, smoke.target_cache_key(flags + list(override), profile, self.parser))
        from dataclasses import replace
        self.assertNotEqual(key, smoke.target_cache_key(flags, replace(profile, data_limit=200), self.parser))

    def test_edge_filters_and_warmup_follow_requested_epochs(self):
        _, cases = self.cases(["--preset", "edge", "--epochs", "1", "--profiles", "alternate",
                               "--only", "defense:weighted_smoothing", "attack:quantile"])
        self.assertEqual(len(cases), 2)
        for case in cases:
            settings = self.settings(case)
            self.assertEqual(settings.defender_epochs, 1)
            self.assertLess(settings.defender_weighted_smoothing_warmup_epochs, 1)
        _, cases = self.cases(["--preset", "edge", "--grid", "defender_lr=0.01,0.02",
                               "--only", "attack:pmia_loss"])
        self.assertEqual(len(cases), 10)
        self.assertEqual(len({self.settings(c).out_folder for c in cases}), 2)


if __name__ == "__main__":
    unittest.main()

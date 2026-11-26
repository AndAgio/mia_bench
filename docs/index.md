# mia_bench

mia_bench is a benchmark for membership inference attacks and defenses, in which any attack can be run against any
defense. To install it and run a first experiment, see the [README](https://github.com/AndAgio/mia_bench#readme).

- [Defenses](defenses/index.md)
- [Attacks](attacks/index.md), and the [decision rules](attacks/decision_rules.md) that turn their scores into
  decisions, by default the same for every attack or, with `--decision_rule attack`, as in each attack's paper
- [Command-line options](reference/cli.md)
- [Training defaults](reference/training.md)
- [API reference](reference/api.md)
- [Documenting a method](contributing.md)

```{toctree}
:hidden:
:caption: Methods

defenses/index
attacks/index
```

```{toctree}
:hidden:
:caption: Reference

reference/cli
reference/training
reference/api
```

```{toctree}
:hidden:
:caption: Contributing

contributing
```

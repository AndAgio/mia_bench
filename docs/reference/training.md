# Training defaults

The target model trains with the `--defender_*` options and the shadow models with the `--attacker_*` ones. Seven of
them have a default for each dataset: the optimizer, the learning rate, its schedule, the weight decay, the momentum,
the number of epochs and the batch size. An option left unset takes its dataset's value from this table, generated
from `TRAINING_DEFAULTS` in `src/utils/training_defaults.py`. A value given on the command line always wins.

```{include} ../_generated/training_defaults.md
```

The default is filled in when the options are parsed, so it is part of the run's hash like a value given on the
command line. Changing a default in the table therefore retrains the models of the runs it affects, and only those.

## How the defaults were chosen

The image datasets keep the benchmark's original defaults for its ResNets.

The tabular datasets use the tabular MLP, which plain SGD at a learning rate of 0.01 barely trains: on Texas, it
reached 22.6% train and 22.5% test accuracy after 30 epochs, so a model trained this way memorizes almost nothing for
an attack to find. We compared two recipes on 20,000 training records of each dataset (5,000 on News, which has
fewer): AdamW at 0.001 for 50 epochs, and the RMIA authors' Purchase recipe, SGD at 0.1 without momentum or weight
decay for 100 epochs, here with a cosine schedule. Each dataset takes the recipe that fits its training data with the
better test accuracy. Train and test accuracy at the last epoch:

| Dataset | AdamW | SGD, the authors' Purchase recipe |
|---|---|---|
| Purchase | 99.8% / 75.6% | 99.7% / 82.7% |
| Texas | 100% / 55.0% | 74.5% / 55.6% |
| News | 100% / 76.8% | 5.7% / 5.3% |

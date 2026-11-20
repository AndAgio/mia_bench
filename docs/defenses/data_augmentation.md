# Data augmentation

Data augmentation trains on random transformations of the training samples, so the model sees each sample in a
different form at every epoch and memorizes the samples themselves less. It has no source paper of its own:
[Kaya and Dumitras (ICML 2021)](https://proceedings.mlr.press/v139/kaya21a.html) and
[Yu et al. (2020)](https://arxiv.org/abs/2007.10567) evaluate augmentations of images as membership inference
defenses. Inference is unchanged.

Images go through a random crop after padding (off by default), a horizontal flip, a random rotation and the erasing
of a random rectangle. Tabular datasets have no images to transform, so each feature of a training sample is instead
set to 0 at random: a masked feature of Purchase or Texas reads as an item not bought or a procedure not done, and one
of News as an absent word. Bit flips (Purchase and Texas only) and Gaussian noise can be added too, and are off by
default.

## Usage

```bash
python run.py --defender_mode data_augmentation --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.data_augmentation.DataAugmentationDefender`.

## Parameters

`--defender_augment_crop_padding` pads each image by this many pixels and crops it back to its size at a random
position; a padding of 4 with flips is the standard augmentation on CIFAR. `--defender_augment_horizontal_flip_prob`
and `--defender_augment_erase_prob` are the probabilities of a flip and of an erasing, and
`--defender_augment_rotation` the largest angle of the rotations. 0 turns each of them off.

On tabular datasets, `--defender_augment_feature_mask_prob` is the probability of masking each feature,
`--defender_augment_feature_flip_prob` that of flipping each binary feature, and
`--defender_augment_feature_noise_std` the standard deviation of the Gaussian noise added to each feature. News rows
have unit norm over 134,410 dimensions, so useful noise values are small: 0.01 already gives noise of norm 3.7.
Setting a tabular option on an image dataset is an error.

```{include} ../_generated/data_augmentation_parameters.md
```

The default masking probability lowered a threshold attack on all three tabular datasets in our check, and on Texas,
where the model overfits most, it also raised the test accuracy.

```{include} ../_generated/options_data_augmentation.md
```

## Limitations

- Attacks that train shadow models train them without augmentation, unlike the target
  ([issue #48](https://github.com/AndAgio/mia_bench/issues/48)).
- Augmentation cannot be combined with another defense yet
  ([issue #47](https://github.com/AndAgio/mia_bench/issues/47)).

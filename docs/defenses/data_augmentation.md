# Data augmentation

Data augmentation trains on random transformations of the training samples, so the model sees each sample in a
different form at every epoch and memorizes the samples themselves less. It has no source paper of its own:
[Kaya and Dumitras (ICML 2021)](https://proceedings.mlr.press/v139/kaya21a.html) and
[Yu et al. (2020)](https://arxiv.org/abs/2007.10567) evaluate augmentations of images as membership inference
defenses. Each training image goes through, in order, a random crop after padding (off by default), a horizontal
flip, a random rotation and the erasing of a random rectangle: torchvision's RandomCrop, RandomHorizontalFlip,
RandomRotation and RandomErasing. The transforms act on normalized images, so the padding, the corners of rotated
images and the erased rectangles take the dataset's mean colour.

Tabular datasets have no images to flip or rotate, so the image transforms are skipped there. Each feature of a
training sample is instead set to 0 at random, the counterpart of erasing for feature vectors: a masked feature of
Purchase or Texas reads as an item not bought or a procedure not done, and one of News as an absent word. Bit flips
(Purchase and Texas only) and Gaussian noise can be applied before the masking, and are off by default.

Inference is unchanged, and the training accuracy in the results is measured on the samples themselves.

## Usage

```bash
python run.py --defender_mode data_augmentation --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.data_augmentation.DataAugmentationDefender`.

## Parameters

`--defender_augment_crop_padding` pads each image by this many pixels and crops it back to its size at a random
position. Crops with a padding of 4, together with flips, are the standard augmentation on CIFAR.
`--defender_augment_horizontal_flip_prob` and `--defender_augment_erase_prob` are the probabilities of a flip and of
an erasing, and `--defender_augment_rotation` the largest angle of the rotations. 0 turns each of them off.

On tabular datasets, `--defender_augment_feature_mask_prob` is the probability of masking each feature,
`--defender_augment_feature_flip_prob` that of flipping each binary feature (Purchase and Texas only), and
`--defender_augment_feature_noise_std` the standard deviation of the Gaussian noise added to each feature, meant for
the continuous features of News. News rows have unit norm over 134,410 dimensions, so useful values are small: a
standard deviation of 0.01 already gives noise of norm 3.7. Setting any of these options on an image dataset is an
error.

```{include} ../_generated/data_augmentation_parameters.md
```

We chose the masking probability with a check on each tabular dataset: the tabular MLP, trained with Adam for 50
epochs on 10,000 samples (5,000 on News, which has fewer), with the model of the best validation accuracy, and a
Mentr threshold attack against as many other samples. Each row is a single run, so differences of 2 to 3 points may
come from the choice of the best epoch. Masking with a probability of 0.2 lowers the attack on all three datasets,
and on Texas, where the model overfits most, it also raises the test accuracy. Bit flips do even better on Texas but
not consistently on Purchase, and Gaussian noise does no better than masking on News, so both are off by default.

| Dataset | Augmentation | Train accuracy | Test accuracy | Attack accuracy |
|---|---|---|---|---|
| Purchase | None | 97.9% | 72.0% | 68.0% |
| Purchase | Masking, p = 0.1 | 95.2% | 70.0% | 66.2% |
| Purchase | Masking, p = 0.2 | 93.4% | 69.4% | 64.8% |
| Purchase | Masking, p = 0.3 | 84.1% | 62.9% | 61.4% |
| Purchase | Bit flips, p = 0.01 | 95.6% | 71.5% | 65.8% |
| Purchase | Bit flips, p = 0.02 | 99.2% | 72.3% | 71.3% |
| Purchase | Bit flips, p = 0.05 | 95.8% | 70.1% | 66.9% |
| Texas | None | 99.9% | 51.5% | 80.8% |
| Texas | Masking, p = 0.1 | 92.5% | 54.1% | 70.6% |
| Texas | Masking, p = 0.2 | 88.5% | 54.7% | 68.6% |
| Texas | Masking, p = 0.3 | 83.2% | 54.5% | 65.3% |
| Texas | Bit flips, p = 0.01 | 85.5% | 55.0% | 66.2% |
| Texas | Bit flips, p = 0.02 | 68.8% | 52.9% | 58.9% |
| News | None | 100% | 84.7% | 79.2% |
| News | Masking, p = 0.1 | 100% | 84.6% | 78.4% |
| News | Masking, p = 0.2 | 100% | 83.3% | 75.8% |
| News | Masking, p = 0.3 | 100% | 83.4% | 77.8% |
| News | Gaussian noise, std = 0.0005 | 100% | 82.6% | 77.9% |
| News | Gaussian noise, std = 0.001 | 100% | 82.5% | 78.3% |
| News | Gaussian noise, std = 0.002 | 100% | 82.4% | 75.6% |

```{include} ../_generated/options_data_augmentation.md
```

## Limitations

- Attacks that train shadow models train them without augmentation, unlike the target
  ([issue #48](https://github.com/AndAgio/mia_bench/issues/48)), so the shadow models differ from the target model
  more than these attacks assume.
- Augmentation cannot be combined with another defense yet
  ([issue #47](https://github.com/AndAgio/mia_bench/issues/47)).

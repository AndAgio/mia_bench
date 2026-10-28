# API reference

The classes to use or extend when adding an attack or a defense. Their methods and signatures are read from the code.

## Extending

```{eval-rst}
.. autoapiclass:: src.mia.defenses.base.BaseDefender
   :members:

.. autoapiclass:: src.mia.defenses.base.DefendedModel
   :members:

.. autoapiclass:: src.mia.attacks.base_mia.BaseMIA
   :members:

.. autoapiclass:: src.trainer.train_manager.TrainManager
   :members:
```

## Defenses

```{eval-rst}
.. autoapiclass:: src.mia.defenses.vanilla.VanillaDefender
.. autoapiclass:: src.mia.defenses.training.differential_privacy.DifferentialPrivacyDefender
.. autoapiclass:: src.mia.defenses.training.mixup.MixupDefender
.. autoapiclass:: src.mia.defenses.training.data_augmentation.DataAugmentationDefender
.. autoapiclass:: src.mia.defenses.training.adversarial_regularization.AdvRegDefender
.. autoapiclass:: src.mia.defenses.training.relax_loss.RelaxLossDefender
.. autoapiclass:: src.mia.defenses.training.selena.SelenaDefender
.. autoapiclass:: src.mia.defenses.training.mist.MistDefender
.. autoapiclass:: src.mia.defenses.training.weigthed_smoothing.WeightedSmoothingDefender
.. autoapiclass:: src.mia.defenses.training.mmd.MmdDefender
.. autoapiclass:: src.mia.defenses.post_hoc.mem_guard.MemGuardDefender
.. autoapiclass:: src.mia.defenses.post_hoc.ldl.LdlDefender
.. autoapiclass:: src.mia.defenses.post_hoc.purifier.PurifierDefender
.. autoapiclass:: src.mia.defenses.hybrid.hamp.HampDefender
```

## Attacks

```{eval-rst}
.. autoapiclass:: src.mia.attacks.black_box.lira.LiRA
.. autoapiclass:: src.mia.attacks.black_box.robust.RobustMIA
.. autoapiclass:: src.mia.attacks.black_box.attack_p.AttackPMIA
.. autoapiclass:: src.mia.attacks.black_box.attack_r.AttackRMIA
.. autoapiclass:: src.mia.attacks.black_box.quantile.QuantileMIA
.. autoapiclass:: src.mia.attacks.black_box.neural.NeuralMIA
.. autoapiclass:: src.mia.attacks.label_only.noise_robustness.NoiseRobustnessMIA
.. autoapiclass:: src.mia.attacks.label_only.supervised_boundary.SupervisedBoundaryMIA
.. autoapiclass:: src.mia.attacks.label_only.unsupervised_boundary.UnsupervisedBoundaryMIA
.. autoapiclass:: src.mia.attacks.label_only.transfer.TransferMIA
.. autoapiclass:: src.mia.attacks.label_only.oslo.OsloMIA
.. autoapiclass:: src.mia.attacks.label_only.dh_attack.DHAttack
.. autoapiclass:: src.mia.attacks.label_only.yoqo.Yoqo
```

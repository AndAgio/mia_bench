from src.mia.attacks.black_box.robust import RobustMIA
from src.mia.attacks.black_box.lira import LiRA
from src.mia.attacks.black_box.quantile import QuantileMIA
from src.mia.attacks.black_box.neural import NeuralMIA
from src.mia.attacks.black_box.attack_r import AttackRMIA
from src.mia.attacks.black_box.attack_p import AttackPMIA
from src.mia.attacks.label_only.supervised_boundary import SupervisedBoundaryMIA
from src.mia.attacks.label_only.unsupervised_boundary import UnsupervisedBoundaryMIA
from src.mia.attacks.label_only.noise_robustness import NoiseRobustnessMIA
from src.mia.attacks.label_only.transfer import TransferMIA
from src.mia.attacks.label_only.oslo import OsloMIA
from src.mia.attacks.label_only.dh_attack import DHAttack
from src.mia.attacks.label_only.yoqo import Yoqo


def get_attacker_class(attack_mode: str):
    if attack_mode in ['online_robust', 'offline_robust', 'on_robust', 'off_robust']:
        return RobustMIA
    elif attack_mode == 'lira':
        return LiRA
    elif attack_mode == 'quantile':
        return QuantileMIA
    elif attack_mode in ['neural_feat', 'neural_prob', 'neural_logit']:
        return NeuralMIA
    elif attack_mode in ['rmia_loss', 'rmia_confidence', 'rmia_entropy']:
        return AttackRMIA
    elif attack_mode in ['pmia_loss', 'pmia_confidence', 'pmia_entropy']:
        return AttackPMIA
    elif attack_mode in ['sba', 'sba_hopskipjump', 'sba_hsj', 'sba_hopskip', 'sba_hop', 'sba_qeba', 'sba_qeba-spatial', 'sba_qeba-dct', 'sba_qeba-pca', 'sba_qeba-custom']:
        return SupervisedBoundaryMIA
    elif attack_mode in ['uba', 'uba_hopskipjump', 'uba_hsj', 'uba_hopskip', 'uba_hop', 'uba_qeba', 'uba_qeba-spatial', 'uba_qeba-dct', 'uba_qeba-pca', 'uba_qeba-custom']:
        return UnsupervisedBoundaryMIA
    elif attack_mode in ['noise_robust', 'noise_robustness', 'noise_rob', 'nr']:
        return NoiseRobustnessMIA
    elif attack_mode in ['transfer_loss', 'transfer_confidence', 'transfer_entropy']:
        return TransferMIA
    elif attack_mode in ['oslo', 'oslo_difgsm', 'oslo_mifgsm', 'oslo_tifgsm', 'oslo_tmifgsm']:
        return OsloMIA
    elif attack_mode in ['dh', 'dh_white', 'dh_black', 'dh_random', 'dh-attack', 'dh-attack_white', 'dh-attack_black', 'dh-attack_random',]:
        return DHAttack
    elif attack_mode in ['online_yoqo', 'offline_yoqo', 'on_yoqo', 'off_yoqo', 'yoqo']:
        return Yoqo
    else:
        raise ValueError('Attack "{}" not found or not implemented yet! Double check your settings please!'.format(attack_mode))
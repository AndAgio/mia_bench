from src.mia.attacks.black_box.robust import RobustMIA
from src.mia.attacks.black_box.lira import LiRA
from src.mia.attacks.black_box.quantile import QuantileMIA
from src.mia.attacks.black_box.neural import NeuralMIA
from src.mia.attacks.black_box.attack_r import AttackRMIA
from src.mia.attacks.black_box.attack_p import AttackPMIA


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
    else:
        raise ValueError('Attack "{}" not found or not implemented yet! Double check your settings please!'.format(attack_mode))
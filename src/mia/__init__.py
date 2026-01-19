from src.mia.rmia import RMIA
from src.mia.lira import LiRA
from src.mia.quantile import QuantileMIA
from src.mia.neural import NeuralMIA


def get_attacker_class(attack_mode: str):
    if attack_mode in ['online_rmia', 'offline_rmia', 'on_rmia', 'off_rmia']:
        return RMIA
    elif attack_mode == 'lira':
        return LiRA
    elif attack_mode == 'quantile':
        return QuantileMIA
    elif attack_mode in ['neural_feat', 'neural_prob', 'neural_logit']:
        return NeuralMIA
    else:
        raise ValueError('Attack "{}" not found or not implemented yet! Double check your settings please!'.format(attack_mode))
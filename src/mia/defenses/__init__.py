from src.mia.defenses.vanilla import VanillaVictim
from src.mia.defenses.training.differential_privacy import DifferentialPrivacyDefender



def get_defender_class(defense_mode: str):
    if defense_mode in ['none', 'no', 'vanilla']:
        return VanillaVictim
    elif defense_mode == 'dp':
        return DifferentialPrivacyDefender
    else:
        raise ValueError('Defense "{}" not found or not implemented yet! Double check your settings please!'.format(defense_mode))
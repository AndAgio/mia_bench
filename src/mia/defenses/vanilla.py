from src.mia.defenses.base import BaseDefender
from src.utils.configs import DefenderConfigs, NoDefenseConfigs


class VanillaDefender(BaseDefender):
    def __init__(self, defender_configs: DefenderConfigs):
        assert isinstance(defender_configs.defense, NoDefenseConfigs), f"VanillaDefender can only be used with NoDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'vanilla_defender'
        self.no_defense_configs = defender_configs.defense

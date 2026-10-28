from typing import Union
import torch
from src.mia.defenses.base import BaseDefender
from src.utils.configs import DefenderConfigs, NoDefenseConfigs


class VanillaDefender(BaseDefender):
    """No defense: standard training, and the trained model answers queries unchanged."""
    def __init__(self, defender_configs: DefenderConfigs):
        """Check that the configs select no defense."""
        assert isinstance(defender_configs.defense, NoDefenseConfigs), f"VanillaDefender can only be used with NoDefenseConfigs, got {type(defender_configs.defense)}"
        super().__init__(defender_configs=defender_configs)
        self.name = 'vanilla_defender'
        self.no_defense_configs = defender_configs.defense

    def defend_model(self, device: Union[str, torch.device]) -> torch.nn.Module:
        """Return the trained model unchanged."""
        self.logger.print_it('Vanilla Defender: returning trained model as defended model...')
        self.defended_model = self.trained_model
        return self.defended_model

import math
from sys import stderr
import numpy as np
from dataclasses import dataclass
from .cargo import UnitCargo
from .config import EnvConfig

@dataclass
class Factory:
    team_id: int
    unit_id: str
    strain_id: int
    power: int
    cargo: UnitCargo
    pos: np.ndarray
    lichen_count: int
    # lichen tiles connected to this factory
    # lichen_tiles: np.ndarray
    env_cfg: EnvConfig

    @property
    def pos_slice(self):
        x,y = self.pos
        return slice(x - 1, x + 2), slice(y - 1, y + 2)

    def _heavy_robot_metal_cost(self):
        return self.env_cfg.ROBOTS["HEAVY"].METAL_COST

    def _heavy_robot_power_cost(self):
        return self.env_cfg.ROBOTS["HEAVY"].POWER_COST

    def can_build_heavy(self):
        return self.power >= self._heavy_robot_power_cost() \
            and self.cargo.metal >= self._heavy_robot_metal_cost()

    def build_heavy(self):
        return 1

    def _light_robot_metal_cost(self):
        return self.env_cfg.ROBOTS["LIGHT"].METAL_COST

    def _light_robot_power_cost(self):
        return self.env_cfg.ROBOTS["LIGHT"].POWER_COST

    def can_build_light(self):
        return self.power >= self._light_robot_power_cost() \
            and self.cargo.metal >= self._light_robot_metal_cost()

    def build_light(self):
        return 0

    def _lichen_water_cost(self, game_state):
        """
        Water required to perform water action
        """
        owned_lichen_tiles = (game_state.board.lichen_strains == self.strain_id).sum()
        return np.ceil(owned_lichen_tiles / self.env_cfg.LICHEN_WATERING_COST_FACTOR)

    def can_water_lichen(self, game_state):
        return self.cargo.water >= self._lichen_water_cost(game_state)

    def water_lichen(self):
        return 2
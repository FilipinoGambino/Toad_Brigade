from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from .cargo import UnitCargo
from .config import EnvConfig
from .player import Player
from .robot import Robot
from .factory import Factory

def obs_to_game_state(env_cfg: EnvConfig, obs: Dict[str, Any]):
    units = dict()
    for agent in obs["units"]:
        units[agent] = dict()
        for unit_id in obs["units"][agent]:
            unit_data = obs["units"][agent][unit_id]
            unit = Robot(
                **unit_data,
                unit_cfg=env_cfg.ROBOTS[unit_data["unit_type"]],
                env_cfg=env_cfg,
                only_power=True
            )
            unit.cargo = UnitCargo(**unit_data["cargo"])
            units[agent][unit_id] = unit

    factory_occupancy_map = np.full_like(obs["board"]["rubble"], fill_value=-1, dtype=int)
    power_map = np.zeros_like(obs["board"]["rubble"], dtype=int)
    factories = dict()
    for agent in obs["factories"]:
        factories[agent] = dict()
        for unit_id in obs["factories"][agent]:
            f_data = obs["factories"][agent][unit_id]
            factory = Factory(
                **f_data,
                env_cfg=env_cfg
            )
            factory.cargo = UnitCargo(**f_data["cargo"])
            factories[agent][unit_id] = factory
            factory_occupancy_map[factory.pos_slice] = factory.strain_id
            power_map[factory.pos_slice] = factory.power

    players = dict()
    for agent in obs["teams"]:
        team_data = obs["teams"][agent]
        # team_data['factories_count'] =
        # faction = FactionTypes[team_data["faction"]]
        players[agent] = Player(**team_data, agent=agent)

    lichen_spreading = np.logical_and(
        env_cfg.MIN_LICHEN_TO_SPREAD <= obs["board"]["lichen"],
        obs["board"]["lichen"] < env_cfg.MAX_LICHEN_PER_TILE,
        dtype=float,
    )

    game_state = GameState(
        env_cfg=env_cfg,
        real_env_steps=obs['real_env_steps'],
        board=Board(
            rubble=obs["board"]["rubble"],
            ice=obs["board"]["ice"],
            ore=obs["board"]["ore"],
            lichen=obs["board"]["lichen"],
            lichen_strains=obs["board"]["lichen_strains"],
            lichen_spreading=lichen_spreading,
            power=power_map,
            factory_occupancy_map=factory_occupancy_map,
            factories_per_team=obs["board"]["factories_per_team"],
            lichen_per_team=0,
            valid_spawns_mask=obs["board"]["valid_spawns_mask"]
        ),
        units=units,
        factories=factories,
        players=players,
    )

    return game_state


@dataclass
class Board:
    rubble: np.ndarray
    ice: np.ndarray
    ore: np.ndarray
    lichen: np.ndarray
    lichen_strains: np.ndarray
    lichen_spreading: np.ndarray
    power: np.ndarray
    factory_occupancy_map: np.ndarray
    factories_per_team: int
    lichen_per_team: int
    valid_spawns_mask: np.ndarray

    @property
    def board_sum(self):
        resource_maps = np.stack([self.rubble, self.ore, self.ice, self.lichen], axis=0)
        board_sum = np.sum(resource_maps, axis=0, keepdims=False)
        return board_sum


@dataclass
class GameState:
    """
    A GameState object at step env_steps. Copied from luxai_s2/state/state.py
    """
    real_env_steps: int
    env_cfg: EnvConfig
    board: Board
    units: Dict[str, Dict[str, Robot]] = field(default_factory=dict)
    factories: Dict[str, Dict[str, Factory]] = field(default_factory=dict)
    players: Dict[str, Player] = field(default_factory=dict)

    # @property
    # def real_env_steps(self):
    #     """
    #     the actual env step in the environment, which subtracts the time spent bidding and placing factories
    #     """
    #     if self.env_cfg.BIDDING_SYSTEM:
    #         # + 1 for extra factory placement and + 1 for bidding step
    #         return self.env_steps - (self.board.factories_per_team * 2 + 1)
    #     else:
    #         return self.env_steps

    def is_day(self):
        return self.real_env_steps % self.env_cfg.CYCLE_LENGTH < self.env_cfg.DAY_LENGTH

    def cycle_step(self):
        return self.real_env_steps % self.env_cfg.CYCLE_LENGTH

    def game_phase(self):
        return self.real_env_steps // 100

    # @property
    # def lichen_count(self):
    #     for player_id in self.players:


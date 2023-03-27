from dataclasses import dataclass, field
from typing import Dict
import numpy as np
from .cargo import UnitCargo
from .config import EnvConfig
from .team import Team, FactionTypes
from .unit import Unit
from .factory import Factory

def obs_to_game_state(step, env_cfg: EnvConfig, obs):
    units = dict()
    for agent in obs["units"]:
        units[agent] = dict()
        for unit_id in obs["units"][agent]:
            unit_data = obs["units"][agent][unit_id]
            cargo = UnitCargo(**unit_data["cargo"])
            unit = Unit(
                **unit_data,
                unit_cfg=env_cfg.ROBOTS[unit_data["unit_type"]],
                env_cfg=env_cfg
            )
            unit.cargo = cargo
            units[agent][unit_id] = unit

    factory_occupancy_map = np.ones_like(obs["board"]["rubble"], dtype=int) * -1
    factories = dict()
    for agent in obs["factories"]:
        factories[agent] = dict()
        for unit_id in obs["factories"][agent]:
            f_data = obs["factories"][agent][unit_id]
            cargo = UnitCargo(**f_data["cargo"])
            factory = Factory(
                **f_data,
                env_cfg=env_cfg
            )
            factory.cargo = cargo
            factories[agent][unit_id] = factory
            factory_occupancy_map[factory.pos_slice] = factory.strain_id
    teams = dict()
    for agent in obs["teams"]:
        team_data = obs["teams"][agent]
        faction = FactionTypes[team_data["faction"]]
        teams[agent] = Team(**team_data, agent=agent)

    game_state = GameState(
        env_cfg=env_cfg,
        env_steps=step,
        board=Board(
            rubble=obs["board"]["rubble"],
            ice=obs["board"]["ice"],
            ore=obs["board"]["ore"],
            lichen=obs["board"]["lichen"],
            lichen_strains=obs["board"]["lichen_strains"],
            lichen_spreading=env_cfg.MIN_LICHEN_TO_SPREAD <= obs["board"]["lichen"] < env_cfg.MAX_LICHEN_PER_TILE,
            factory_occupancy_map=factory_occupancy_map,
            factories_per_team=obs["board"]["factories_per_team"],
            valid_spawns_mask=obs["board"]["valid_spawns_mask"]
        ),
        units=units,
        factories=factories,
        teams=teams
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
    factory_occupancy_map: np.ndarray
    factories_per_team: int
    valid_spawns_mask: np.ndarray


@dataclass
class GameState:
    """
    A GameState object at step env_steps. Copied from luxai_s2/state/state.py
    """
    env_steps: int
    env_cfg: dict
    board: Board
    units: Dict[str, Dict[str, Unit]] = field(default_factory=dict)
    factories: Dict[str, Dict[str, Factory]] = field(default_factory=dict)
    teams: Dict[str, Team] = field(default_factory=dict)

    @property
    def real_env_steps(self):
        """
        the actual env step in the environment, which subtracts the time spent bidding and placing factories
        """
        if self.env_cfg.BIDDING_SYSTEM:
            # + 1 for extra factory placement and + 1 for bidding step
            return self.env_steps - (self.board.factories_per_team * 2 + 1)
        else:
            return self.env_steps

    # various utility functions
    def is_day(self):
        return self.real_env_steps % self.env_cfg.CYCLE_LENGTH < self.env_cfg.DAY_LENGTH

    def cycle_step(self):
        return self.real_env_steps % self.env_cfg.CYCLE_LENGTH

    def game_phase(self):
        return self.real_env_steps // 100

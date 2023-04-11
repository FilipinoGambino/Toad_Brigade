from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from .cargo import UnitCargo
from .config import EnvConfig
from .player import Player
from .robot import Robot
from .factory import Factory

def obs_to_game_state(env_cfg: EnvConfig, obs: Dict[str, Any]):
    allegiance_map = np.full_like(obs['board']['rubble'], fill_value=-1)
    robot_map = np.zeros_like(obs['board']['rubble'])
    robot_weight_map = np.zeros_like(obs['board']['rubble'])
    power_light_map = np.zeros_like(obs["board"]["rubble"], dtype=int)
    power_heavy_map = np.zeros_like(obs["board"]["rubble"], dtype=int)
    cargo_light_full = np.zeros_like(obs["board"]["rubble"], dtype=int)
    cargo_heavy_full = np.zeros_like(obs["board"]["rubble"], dtype=int)
    cargo_light_ice = np.zeros_like(obs["board"]["rubble"], dtype=int)
    cargo_light_ore = np.zeros_like(obs["board"]["rubble"], dtype=int)
    cargo_heavy_ice = np.zeros_like(obs["board"]["rubble"], dtype=int)
    cargo_heavy_ore = np.zeros_like(obs["board"]["rubble"], dtype=int)

    units = dict()
    for agent in obs["units"]:
        units[agent] = dict()
        for unit_id in obs["units"][agent]:
            unit_data = obs["units"][agent][unit_id]
            unit_data['pos'] = np.flip(unit_data['pos'])
            unit = Robot(
                **unit_data,
                unit_cfg=env_cfg.ROBOTS[unit_data["unit_type"]],
                env_cfg=env_cfg,
                only_power=True
            )
            unit.cargo = UnitCargo(**unit_data["cargo"])
            units[agent][unit_id] = unit
            row,col = unit.pos
            robot_map[row, col] += 1
            allegiance_map[row, col] = unit.team_id
            if unit_data['unit_type'] == 'LIGHT':
                robot_weight_map[row, col] = 1
                power_light_map[row, col] = unit.power
                cargo_light_full[row, col] = unit.is_full
                cargo_light_ice[row, col] = unit.cargo.ice
                cargo_light_ore[row, col] = unit.cargo.ore
            else:
                robot_weight_map[row, col] = 2
                power_heavy_map[row, col] = unit.power
                cargo_heavy_full[row, col] = unit.is_full
                cargo_heavy_ice[row, col] = unit.cargo.ice
                cargo_heavy_ore[row, col] = unit.cargo.ore

    factory_lichen_count = dict()
    factory_strains = np.full_like(obs["board"]["rubble"], fill_value=-1, dtype=int)
    power_factory_map = np.zeros_like(obs["board"]["rubble"], dtype=int)
    cargo_factory_ice = np.zeros_like(obs["board"]["rubble"], dtype=int)
    cargo_factory_ore = np.zeros_like(obs["board"]["rubble"], dtype=int)
    cargo_factory_water = np.zeros_like(obs["board"]["rubble"], dtype=int)
    cargo_factory_metal = np.zeros_like(obs["board"]["rubble"], dtype=int)

    factories = dict()
    for agent in obs["factories"]:
        factories[agent] = dict()
        for unit_id in obs["factories"][agent]:
            f_data = obs["factories"][agent][unit_id]
            f_data['pos'] = np.flip(f_data['pos'])
            strain_id = f_data['strain_id']

            factory_lichen_count[strain_id] = np.sum(
                obs['board']['lichen'],
                where=obs['board']['lichen_strains'] == strain_id
            )

            factory = Factory(
                **f_data,
                env_cfg=env_cfg,
                lichen_count = factory_lichen_count[strain_id]
            )
            factory.cargo = UnitCargo(**f_data["cargo"])
            factories[agent][unit_id] = factory
            factory_strains[factory.pos_slice] = factory.strain_id
            power_factory_map[factory.pos_slice] = factory.power
            cargo_factory_ice[factory.pos_slice] = factory.cargo.ice
            cargo_factory_ore[factory.pos_slice] = factory.cargo.ore
            cargo_factory_water[factory.pos_slice] = factory.cargo.water
            cargo_factory_metal[factory.pos_slice] = factory.cargo.metal
            allegiance_map[factory.pos_slice] = factory.team_id

    players = dict()
    for agent in obs["teams"]:
        team_data = obs["teams"][agent]
        # team_data['factories_count'] =
        # faction = FactionTypes[team_data["faction"]]
        team_lichen_count = sum(
            [
                lichen_count
                for strain_id, lichen_count in factory_lichen_count.items()
                if strain_id in team_data['factory_strains']
            ]
        )

        players[agent] = Player(
            **team_data,
            agent=agent,
            lichen_count=team_lichen_count,
            factory_count=len(factories[agent]),
            robot_count=len(units[agent])
        )

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
            power_light=power_light_map,
            power_heavy=power_heavy_map,
            power_factory=power_factory_map,
            robots=robot_map,
            robot_weight=robot_weight_map,
            factory_strains=factory_strains,
            allegiance=allegiance_map,
            cargo_light_full=cargo_light_full,
            cargo_heavy_full=cargo_heavy_full,
            cargo_light_ice=cargo_light_ice,
            cargo_light_ore=cargo_light_ore,
            cargo_heavy_ice=cargo_heavy_ice,
            cargo_heavy_ore=cargo_heavy_ore,
            cargo_factory_ice=cargo_factory_ice,
            cargo_factory_ore=cargo_factory_ore,
            cargo_factory_water=cargo_factory_water,
            cargo_factory_metal=cargo_factory_metal,
            valid_spawns_mask=obs["board"]["valid_spawns_mask"]
        ),
        units=units,
        factories=factories,
        players=players,
        factories_per_team=obs["board"]["factories_per_team"],
    )

    return game_state


@dataclass
class Board:
    # Resources
    rubble: np.ndarray
    ice: np.ndarray
    ore: np.ndarray
    # Lichen info
    lichen: np.ndarray
    lichen_strains: np.ndarray
    lichen_spreading: np.ndarray
    # Robot + Factory info
    allegiance: np.ndarray
    # Factory info
    power_factory: np.ndarray
    factory_strains: np.ndarray
    cargo_factory_ice: np.ndarray
    cargo_factory_ore: np.ndarray
    cargo_factory_water: np.ndarray
    cargo_factory_metal: np.ndarray
    # Robot info
    power_light: np.ndarray
    power_heavy: np.ndarray
    robots: np.ndarray
    robot_weight: np.ndarray
    cargo_light_full: np.ndarray
    cargo_heavy_full: np.ndarray
    cargo_light_ice: np.ndarray
    cargo_light_ore: np.ndarray
    cargo_heavy_ice: np.ndarray
    cargo_heavy_ore: np.ndarray
    # Masks
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
    factories_per_team: np.ndarray = np.array([0,0])

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


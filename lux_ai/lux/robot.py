import math
import sys
from typing import List
import numpy as np
from dataclasses import dataclass
from .cargo import UnitCargo
from .config import EnvConfig, UnitConfig

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
directions = dict(
    center=np.array([0, 0]),
    up=np.array([-1, 0]),
    right=np.array([0, 1]),
    down=np.array([1, 0]),
    left=np.array([0, -1])
)
resources = dict(
    ice = 0,
    ore = 1,
    water = 2,
    metal = 3,
    power = 4,
)


@dataclass
class Robot:
    team_id: int
    unit_id: str
    unit_type: str  # "LIGHT" or "HEAVY"
    pos: np.ndarray
    power: int
    cargo: UnitCargo
    env_cfg: EnvConfig
    unit_cfg: UnitConfig
    action_queue: List
    only_power: bool

    @property
    def agent_id(self):
        if self.team_id == 0: return "player_0"
        return "player_1"

    @property
    def is_full(self):
        space = self.env_cfg.ROBOTS[self.unit_type].CARGO_SPACE
        return bool(self.cargo.sum_cargo == space)

    def on_factory_tile(self, gamestate):
        row, col = self.pos
        if gamestate.board.factory_occupancy_map[row, col] != -1:
            return True
        return False

    def action_queue_cost(self):
        cost = self.env_cfg.ROBOTS[self.unit_type].ACTION_QUEUE_POWER_COST
        return cost

    def move_cost(self, game_state, direction):
        board = game_state.board
        target_pos = self.pos + directions[direction]
        if target_pos[0] < 0 \
                or target_pos[1] < 0 \
                or target_pos[1] >= len(board.rubble) \
                or target_pos[0] >= len(board.rubble[0]):
            print(f"Warning, tried to get move cost for going off the map {target_pos}")
            return np.Inf
        factory_there = board.factory_strains[target_pos[0], target_pos[1]]
        if factory_there not in game_state.players[self.agent_id].factory_strains and factory_there != -1:
            print("Warning, tried to get move cost for going onto an opposition factory")
            return np.Inf
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]

        return math.floor(self.unit_cfg.MOVE_COST + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)

    def move(self, direction, repeat=0, n=1):
        assert direction in directions.keys(), f"{direction} not in {directions}"
        return np.array([0, direction, 0, 0, repeat, n])

    def transfer(self, transfer_direction, transfer_resource, transfer_amount, repeat=0, n=1):
        if self.only_power:
            # TODO Maybe this should be in action space somewhere
            transfer_resource = resources['power']
        assert transfer_resource in resources.values(), f"{transfer_resource} not in {resources}"
        return np.array([1, transfer_direction, transfer_resource, transfer_amount, repeat, n])

    def pickup(self, pickup_resource, pickup_amount, repeat=0, n=1):
        if self.only_power:
            pickup_resource = resources['power']
        assert pickup_resource in resources.values(), f"{pickup_resource} not in {resources}"
        return np.array([2, 0, pickup_resource, pickup_amount, repeat, n])

    def dig_cost(self, game_state):
        return self.unit_cfg.DIG_COST

    def dig(self, repeat=0, n=1):
        return np.array([3, 0, 0, 0, repeat, n])

    def self_destruct_cost(self):
        return self.unit_cfg.SELF_DESTRUCT_COST

    def self_destruct(self, repeat=0, n=1):
        return np.array([4, 0, 0, 0, repeat, n])

    def recharge(self, x, repeat=0, n=1):
        return np.array([5, 0, 0, x, repeat, n])

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.pos}"
        return out
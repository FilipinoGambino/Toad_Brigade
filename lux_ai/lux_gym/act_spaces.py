from abc import ABC, abstractmethod
from functools import lru_cache
import gym
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

from ..lux.utils import direction_to
from ..lux.game_state import obs_to_game_state, GameState, EnvConfig
from luxai_s2.utils import animate
from ..lux.utils import direction_to, my_turn_to_place_factory
from luxai_s2.env import LuxAI_S2

from ..lux.robot import Robot


# The maximum number of actions that can be taken by units sharing a square
# All remaining units take the no-op action
MAX_OVERLAPPING_ACTIONS = 4
# DIRECTIONS = Constants.DIRECTIONS
# RESOURCES = Constants.RESOURCE_TYPES

# (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)
class Action(ABC):
    def __init__(self, act_type: str) -> None:
        self.act_type = act_type
        self.repeat = 0

    def __str__(self):
        print("Action Template")

    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError("")


class FactoryBuildAction(Action):
    def __init__(self, robot: str) -> None:
        unit_type = "LIGHT"
        super().__init__(f"build_{unit_type}_robot") # unit_type.name.lower()
        self.unit_type = unit_type

    def __str__(self) -> str:
        return f"{self.act_type}"

    def __call__(self, **kwargs):
        return 0 if self.unit_type == "LIGHT" else 1


class FactoryWaterAction(Action):
    def __init__(self) -> None:
        super().__init__("water_lichen")

    def __str__(self) -> str:
        return f"{self.act_type}"

    def __call__(self, **kwargs):
        return 2


direction_to_name = ["center", "up", "right", "down", "left"]
resource_to_name = ["ice", "ore", "water", "metal", "power"]

class MoveAction(Action):
    def __init__(self, move_dir: int) -> None:
        super().__init__(f"move_{direction_to_name[move_dir]}")
        self.move_dir = move_dir

    def __str__(self) -> str:
        return f"{self.act_type}"

    def __call__(self, **kwargs):
        return np.array([0, self.move_dir, 0, 0, kwargs['repeat'], kwargs['n']])


# class TransferAction(Action):
#     def __init__(
#         self,
#         transfer_dir: int,
#         resource: int,
#         transfer_amount: int,
#         repeat: int = 0,
#     ) -> None:
#         super().__init__("transfer")
#         # a[2] = R = resource type (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)
#         self.transfer_dir = transfer_dir
#         self.resource = resource
#         self.transfer_amount = transfer_amount
#         self.repeat = repeat
#         self.power_cost = 0
#
#     def __str__(self) -> str:
#         return f"{self.act_type} {resource_to_name[self.resource]} {direction_to_name[self.transfer_dir]} (r: {self.repeat})"
#
#     def __call__(self):
#         return np.array(
#             [
#                 1,
#                 self.transfer_dir,
#                 self.resource,
#                 self.transfer_amount,
#                 self.repeat,
#                 0
#             ]
#         )


class PickupAction(Action):
    '''
    Always pickup to fill capacity.
    Only using power.
    '''
    def __init__(self, resource: int) -> None:
        super().__init__(f"pickup_{resource_to_name[resource]}")
        self.resource_name = resource_to_name[resource]
        self.resource = resource

    def __str__(self) -> str:
        return f"{self.act_type} {self.resource_name} to max capacity for unit type"

    def __call__(self, **kwargs):
        return np.array([2, 0, self.resource, kwargs['pickup_amount'], kwargs['repeat'], kwargs['n']])


class DigAction(Action):
    def __init__(self) -> None:
        super().__init__("dig")

    def __str__(self) -> str:
        return f"{self.act_type}"

    def __call__(self, **kwargs):
        return np.array([3, 0, 0, 0, kwargs['repeat'], kwargs['n']])


class SelfDestructAction(Action):
    def __init__(self) -> None:
        super().__init__("self_destruct")

    def __str__(self) -> str:
        return f"{self.act_type}"

    def __call__(self, **kwargs):
        return np.array([4, 0, 0, 0, kwargs['repeat'], kwargs['n']])


class RechargeAction(Action):
    def __init__(self) -> None:
        super().__init__("recharge_n_steps")

    def __str__(self) -> str:
        return f"{self.act_type} {self.repeat}"

    def __call__(self, **kwargs):
        return np.array([5, 0, 0, 0, kwargs['repeat'], kwargs['n']])


# class BaseActSpace(ABC):
#     @abstractmethod
#     def get_action_space(self, board_dims: Tuple[int, int] = (48, 48)) -> gym.spaces.Dict:
#         pass
#
#     @abstractmethod
#     def process_actions(
#             self,
#             action_tensors_dict: Dict[str, np.ndarray],
#             game_state: Game,
#             board_dims: Tuple[int, int],
#             pos_to_unit_dict: Dict[Tuple, Optional[Unit]]
#     ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
#         pass
#
#     @abstractmethod
#     def get_available_actions_mask(
#             self,
#             game_state: Game,
#             board_dims: Tuple[int, int],
#             pos_to_unit_dict: Dict[Tuple, Optional[Unit]],
#             pos_to_city_tile_dict: Dict[Tuple, Optional[CityTile]]
#     ) -> Dict[str, np.ndarray]:
#         pass
#
#     @staticmethod
#     @abstractmethod
#     def actions_taken_to_distributions(actions_taken: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
#         pass
#
#
# class BasicActionSpace(BaseActSpace):
#     def __init__(self, board_dims: Optional[Tuple[int, int]] = None):
#         self.board_dims = board_dims
#
#     @lru_cache(maxsize=None)
#     def get_action_space(self, board_dims: Optional[Tuple[int, int]] = None) -> gym.spaces.Dict:
#         if board_dims is None:
#             board_dims = self.board_dims
#         x = board_dims
#         y = board_dims
#         # Player count
#         p = 2
#         return gym.spaces.Dict({
#             "robot": gym.spaces.MultiDiscrete(np.zeros((1, p, x, y), dtype=int) + len(ACTION_TO_FUNC["robot"])),
#             "factory": gym.spaces.MultiDiscrete(np.zeros((1, p, x, y), dtype=int) + len(ACTION_TO_FUNC["factory"])
#             ),
#         })
#
#     def process_actions(
#             self,
#             action_tensors_dict: Dict[str, np.ndarray],
#             game_state: Game,
#             board_dim: int,
#             pos_to_unit_dict: Dict[Tuple, Optional[Unit]]
#     ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
#         action_strs = [[], []]
#         actions_taken = {
#             key: np.zeros(space, dtype=bool) for key, space in self.get_action_space(board_dim).items()
#         }
#         for player in game_state.players:
#             p_id = player.team
#             worker_actions_taken_count = np.zeros(board_dim, dtype=int)
#             cart_actions_taken_count = np.zeros_like(worker_actions_taken_count)
#             for unit in player.units:
#                 if unit.can_act():
#                     x, y = unit.pos.x, unit.pos.y
#                     if unit.is_worker():
#                         unit_type = "worker"
#                         actions_taken_count = worker_actions_taken_count
#                     elif unit.is_cart():
#                         unit_type = "cart"
#                         actions_taken_count = cart_actions_taken_count
#                     else:
#                         raise NotImplementedError(f'New unit type: {unit}')
#                     # Action plane is selected for stacked units
#                     actor_count = actions_taken_count[x, y]
#                     if actor_count >= MAX_OVERLAPPING_ACTIONS:
#                         action = None
#                     else:
#                         action_idx = action_tensors_dict[unit_type][0, p_id, x, y, actor_count]
#                         action_meaning = ACTION_MEANINGS[unit_type][action_idx]
#                         action = get_unit_action(unit, action_idx, pos_to_unit_dict)
#                         action_was_taken = action_meaning == "NO-OP" or (action is not None and action != "")
#                         actions_taken[unit_type][0, p_id, x, y, action_idx] = action_was_taken
#                         # If action is NO-OP, skip remaining actions for units at same location
#                         if action_meaning == "NO-OP":
#                             actions_taken_count[x, y] += MAX_OVERLAPPING_ACTIONS
#                     # None means no-op
#                     # "" means invalid transfer action - fed to game as no-op
#                     if action is not None and action != "":
#                         # noinspection PyTypeChecker
#                         action_strs[p_id].append(action)
#                     actions_taken_count[x, y] += 1
#             for city in player.cities.values():
#                 for city_tile in city.citytiles:
#                     if city_tile.can_act():
#                         x, y = city_tile.pos.x, city_tile.pos.y
#                         action_idx = action_tensors_dict["city_tile"][0, p_id, x, y, 0]
#                         action_meaning = ACTION_MEANINGS["city_tile"][action_idx]
#                         action = get_city_tile_action(city_tile, action_idx)
#                         action_was_taken = action_meaning == "NO-OP" or (action is not None and action != "")
#                         actions_taken["city_tile"][0, p_id, x, y, action_idx] = action_was_taken
#                         # None means no-op
#                         if action is not None:
#                             # noinspection PyTypeChecker
#                             action_strs[p_id].append(action)
#         return action_strs, actions_taken
#
#     def get_available_actions_mask(
#             self,
#             game_state: Game,
#             board_dims: Tuple[int, int],
#             pos_to_unit_dict: Dict[Tuple, Optional[Unit]],
#             pos_to_city_tile_dict: Dict[Tuple, Optional[CityTile]]
#     ) -> Dict[str, np.ndarray]:
#         available_actions_mask = {
#             key: np.ones(space.shape + (len(ACTION_MEANINGS[key]),), dtype=bool)
#             for key, space in self.get_action_space(board_dims).spaces.items()
#         }
#         for player in game_state.players:
#             p_id = player.team
#             for unit in player.units:
#                 if unit.can_act():
#                     x, y = unit.pos.x, unit.pos.y
#                     if unit.is_worker():
#                         unit_type = "worker"
#                     elif unit.is_cart():
#                         unit_type = "cart"
#                     else:
#                         raise NotImplementedError(f"New unit type: {unit}")
#                     # No-op is always a legal action
#                     # Moving is usually a legal action, except when:
#                     #   The unit is at the edge of the board and would try to move off of it
#                     #   The unit would move onto an opposing city tile
#                     #   The unit would move onto another unit with cooldown > 0
#                     # Transferring is only a legal action when:
#                     #   There is an allied unit in the target square
#                     #   The transferring unit has > 0 cargo of the designated resource
#                     #   The receiving unit has cargo space remaining
#                     # Workers: Pillaging is only a legal action when on a road tile and is not on an allied city
#                     # Workers: Building a city is only a legal action when the worker has the required resources and
#                     #       is not on a resource tile
#                     for direction in DIRECTIONS:
#                         new_pos_tuple = unit.pos.translate(direction, 1)
#                         new_pos_tuple = new_pos_tuple.x, new_pos_tuple.y
#                         # Moving and transferring - check that the target position exists on the board
#                         if new_pos_tuple not in pos_to_unit_dict.keys():
#                             available_actions_mask[unit_type][
#                                 :,
#                                 p_id,
#                                 x,
#                                 y,
#                                 ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
#                             ] = False
#                             for resource in RESOURCES:
#                                 available_actions_mask[unit_type][
#                                     :,
#                                     p_id,
#                                     x,
#                                     y,
#                                     ACTION_MEANINGS_TO_IDX[unit_type][f"TRANSFER_{resource}_{direction}"]
#                                 ] = False
#                             continue
#                         # Moving - check that the target position does not contain an opposing city tile
#                         new_pos_city_tile = pos_to_city_tile_dict[new_pos_tuple]
#                         if new_pos_city_tile and new_pos_city_tile.team != p_id:
#                             available_actions_mask[unit_type][
#                                 :,
#                                 p_id,
#                                 x,
#                                 y,
#                                 ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
#                             ] = False
#                         # Moving - check that the target position does not contain a unit with cooldown > 0
#                         new_pos_unit = pos_to_unit_dict[new_pos_tuple]
#                         if new_pos_unit and new_pos_unit.cooldown > 0:
#                             available_actions_mask[unit_type][
#                                 :,
#                                 p_id,
#                                 x,
#                                 y,
#                                 ACTION_MEANINGS_TO_IDX[unit_type][f"MOVE_{direction}"]
#                             ] = False
#                         for resource in RESOURCES:
#                             if (
#                                     # Transferring - check that there is an allied unit in the target square
#                                     (new_pos_unit is None or new_pos_unit.team != p_id) or
#                                     # Transferring - check that the transferring unit has the designated resource
#                                     (unit.cargo.get(resource) <= 0) or
#                                     # Transferring - check that the receiving unit has cargo space
#                                     (new_pos_unit.get_cargo_space_left() <= 0)
#                             ):
#                                 available_actions_mask[unit_type][
#                                     :,
#                                     p_id,
#                                     x,
#                                     y,
#                                     ACTION_MEANINGS_TO_IDX[unit_type][f"TRANSFER_{resource}_{direction}"]
#                                 ] = False
#                     if unit.is_worker():
#                         # Pillaging - check that worker is on a road tile and not on an allied city tile
#                         if game_state.map.get_cell_by_pos(unit.pos).road <= 0 or \
#                                 pos_to_city_tile_dict[(unit.pos.x, unit.pos.y)] is not None:
#                             available_actions_mask[unit_type][
#                                 :,
#                                 p_id,
#                                 x,
#                                 y,
#                                 ACTION_MEANINGS_TO_IDX[unit_type]["PILLAGE"]
#                             ] = False
#                         # Building a city - check that worker has >= the required resources and is not on a resource
#                         if not unit.can_build(game_state.map):
#                             available_actions_mask[unit_type][
#                                 :,
#                                 p_id,
#                                 x,
#                                 y,
#                                 ACTION_MEANINGS_TO_IDX[unit_type]["BUILD_CITY"]
#                             ] = False
#             for city in player.cities.values():
#                 for city_tile in city.citytiles:
#                     if city_tile.can_act():
#                         # No-op is always a legal action
#                         # Research is a legal action whenever research_points < max_research
#                         # Building a new unit is only a legal action when n_units < n_city_tiles
#                         x, y = city_tile.pos.x, city_tile.pos.y
#                         if player.research_points >= MAX_RESEARCH:
#                             available_actions_mask["city_tile"][
#                                 :,
#                                 p_id,
#                                 x,
#                                 y,
#                                 ACTION_MEANINGS_TO_IDX["city_tile"]["RESEARCH"]
#                             ] = False
#                         if len(player.units) >= player.city_tile_count:
#                             available_actions_mask["city_tile"][
#                                 :,
#                                 p_id,
#                                 x,
#                                 y,
#                                 ACTION_MEANINGS_TO_IDX["city_tile"]["BUILD_WORKER"]
#                             ] = False
#                             available_actions_mask["city_tile"][
#                                 :,
#                                 p_id,
#                                 x,
#                                 y,
#                                 ACTION_MEANINGS_TO_IDX["city_tile"]["BUILD_CART"]
#                             ] = False
#         return available_actions_mask
#
#     @staticmethod
#     def actions_taken_to_distributions(actions_taken: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
#         out = {}
#         for space, actions in actions_taken.items():
#             out[space] = {
#                 ACTION_MEANINGS[space][i]: actions[..., i].sum()
#                 for i in range(actions.shape[-1])
#             }
#         return out
#
#
# def get_unit_action(unit: Unit, action_idx: int, pos_to_unit_dict: Dict[Tuple, Optional[Unit]]) -> Optional[str]:
#     if unit.is_worker():
#         unit_type = "worker"
#     elif unit.is_cart():
#         unit_type = "cart"
#     else:
#         raise NotImplementedError(f'New unit type: {unit}')
#     action = ACTION_MEANINGS[unit_type][action_idx]
#     if action.startswith("TRANSFER"):
#         # noinspection PyArgumentList
#         return ACTION_MEANING_TO_FUNC[unit_type][action](unit, pos_to_unit_dict)
#     else:
#         return ACTION_MEANING_TO_FUNC[unit_type][action](unit)
#
#
# def get_city_tile_action(city_tile: CityTile, action_idx: int) -> Optional[str]:
#     action = ACTION_MEANINGS["city_tile"][action_idx]
#     return ACTION_MEANING_TO_FUNC["city_tile"][action](city_tile)

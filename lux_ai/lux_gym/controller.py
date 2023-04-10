import sys
from typing import Any, Dict#, NoneType

import numpy as np
import numpy.typing as npt
from gym import spaces

from abc import ABC

from lux_ai.lux.game_state import GameState

import luxai_s2.unit as luxai_unit
from luxai_s2.unit import UnitType
from luxai_s2.config import EnvConfig
from ..lux_gym.act_spaces import action_to_func

MAP_SIZE = EnvConfig.map_size
ROBOT_ACTIONS = 8 + 1
ROBOTS = EnvConfig().ROBOTS

# Controller class copied here since you won't have access to the luxai_s2 package directly on the competition server
class Controller(ABC):
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = action_space

    def action_to_lux_action(self, agent: str, obs: Dict[str, Any], action: npt.NDArray):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        pass

    def action_masks(self, agent: str, obs: GameState):
        """
        Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not
        """
        pass


class LuxController(Controller):
    def __init__(self, env_cfg, flags) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one HEAVY robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - planning (via actions executing multiple times or repeating actions)
        -

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        self.env_cfg = env_cfg
        self.flags = flags

        actions = self.flags.actions
        self.robot_actions = actions['robot_actions']
        self.factory_actions = actions['factory_actions']

        self.robot_action_id = {
            idx: action_str
            for idx,action_str in enumerate(self.robot_actions)
        }
        self.factory_action_id = {
            idx: action_str
            for idx, action_str in enumerate(self.factory_actions)
        }
        self.robot_funcs = {
            action: action_to_func(action)
            for action in self.robot_actions
        }
        self.factory_funcs = {
            action: action_to_func(action)
            for action in self.factory_actions
        }

        n_factory_actions = len(actions['factory_actions'])
        n_robot_actions = len(actions['robot_actions'])

        n_actions = n_robot_actions + n_factory_actions
        action_space = spaces.Discrete(n_actions)

        super().__init__(action_space)

    def action_to_lux_action(self, agent: str, obs: GameState, actions: Dict[str, npt.NDArray]):
        lux_action = dict()
        action_flag = self.flags.actions

        unit_actions = np.stack(
            [
                action_map.copy()
                for action_key, action_map in actions.items()
                if action_key in action_flag['robot_actions']
            ],
            axis = 0
        )
        factory_actions = np.stack(
            [
                action_map.copy()
                for action_key, action_map in actions.items()
                if action_key in action_flag['factory_actions']
            ],
            axis = 0
        )

        unit_actions = unit_actions.argmax(axis=0)
        factory_actions = factory_actions.argmax(axis=0)
        # print(unit_actions, '\n',factory_actions)

        for unit_id, unit in obs.units[agent].items():
            weight_class = unit.unit_type
            bot_cfg = ROBOTS[weight_class]
            row,col = unit.pos
            if unit.unit_type == 'LIGHT':
                power_space = min(bot_cfg.BATTERY_CAPACITY - unit.power, obs.board.power_light[row,col])
            else:
                power_space = min(bot_cfg.BATTERY_CAPACITY - unit.power, obs.board.power_heavy[row,col])

            action_idx = unit_actions[row,col]
            action = self.robot_action_id[action_idx]

            if action_idx != 0: # Not no_op
                action_queue = [
                    self.robot_funcs[action](
                        pickup_amount=power_space,
                        repeat=0,
                        n=1,
                    )
                ]
                if unit.action_queue:
                    if np.all(unit.action_queue[0] != action_queue[0]):
                        lux_action[unit_id] = action_queue
                else:
                    lux_action[unit_id] = action_queue

        for factory_id, factory in obs.factories[agent].items():
            row, col = factory.pos
            action_idx = factory_actions[row, col]
            action = self.factory_action_id[action_idx]
            lux_action[factory_id] = self.factory_funcs[action](repeat=0, n=1)

        return lux_action

    def action_masks(self, agent: str, obs: GameState):
        """
        Masked Actions:
            Move -> Out-of-bounds / moving onto enemy factory tiles / not enough power
            Transfer -> **For simplicity sake, I'm not including this** (resource + direction adds a few layers)
            Pickup -> Not on ally factory tile (Just power; ice/ore/water/metal irrelavent w/o transfer)
            Dig -> On factory tile / not on resource tile / on ally lichen tile / not enough power
            Self Destruct -> On factory tile / not enough power
            Recharge -> Already at full power capacity
            Repeat -> TODO
        """

        map_shape = (MAP_SIZE, MAP_SIZE)

        # compute a factory occupancy map that will be useful for checking if a board tile
        # has a factory and which team's factory it is.
        action_masks = {
            action: np.ones(map_shape, dtype=bool)
            for action_space in self.flags.actions.keys()
            for action in self.flags.actions[action_space]
        }

        # Out of bounds
        action_masks['move_up'][0, :] = False
        action_masks['move_right'][:, -1] = False
        action_masks['move_down'][-1, :] = False
        action_masks['move_left'][:, 0] = False

        ally_lichen_strains = {player.agent: player.factory_strains for player in obs.players.values()}
        lichen_allegiance_map = np.full(map_shape, dtype=int, fill_value=-1)
        factories_map = np.full(map_shape, dtype=int, fill_value=-1)

        # 0: enemy, 1: ally, -1: N/A
        enemy = 0
        ally = 1
        na = -1
        allegiance = [0, 1, -1]
        # 0: center, 1: up, 2: right, 3: down, 4: left
        move_directions = dict(
            center=np.array([0, 0]),
            up=np.array([-1, 0]),
            right=np.array([0, 1]),
            down=np.array([1, 0]),
            left=np.array([0, -1])
        )

        action_masks['build_light'][:] = False
        action_masks['build_heavy'][:] = False
        action_masks['grow_lichen'][:] = False

        for player_id in obs.players.keys():
            for factory_id, factory in obs.factories[player_id].items():
                row,col = factory.pos
                strain_id = factory.strain_id

                factories_map[ # Factories are always 3x3 so no need to check out-of-bounds
                    factory.pos_slice
                ] = allegiance[agent == player_id]

                lichen_allegiance_coords = np.argwhere(
                    obs.board.lichen_strains == strain_id,
                )
                lichen_allegiance_map[lichen_allegiance_coords] = allegiance[agent == player_id]

                if agent == player_id:
                    if factory.can_build_light():
                        action_masks['build_light'][row, col] = True
                    if factory.can_build_heavy():
                        action_masks['build_heavy'][row, col] = True
                    if factory.can_water_lichen(obs):
                        action_masks['grow_lichen'][row, col] = True

                    # Don't move to the center tile of a factory
                    for direction, (drow, dcol) in move_directions.items():
                        clip_row = max(0, min(row - drow, MAP_SIZE))
                        clip_col = max(0, min(col - dcol, MAP_SIZE))
                        action_masks[f'move_{direction}'][
                            clip_row,
                            clip_col,
                        ] = False

                else:
                    # Masks moving from (row,col) position to illegal position
                    for direction,(drow,dcol) in move_directions.items():
                        action_masks[f'move_{direction}'][
                            max(0, row - 1 - drow): min(row + 2 - drow, MAP_SIZE),
                            max(0, col - 1 - dcol): min(col + 2 - dcol, MAP_SIZE),
                        ] = False

        # pickup action legal only on ally factory tiles
        action_masks['pickup'] = np.where(factories_map == allegiance[ally], True, False)

        # _, index = np.unique(obs.board.lichen_strains, return_inverse=True)
        # print(lichen_allegiance_map[index].reshape(map_shape))
        # ally_lichen_map = lichen_allegiance_map[index].reshape(map_shape)

        board_sum = obs.board.board_sum

        action_masks['dig'] = np.where(
            (factories_map == allegiance[na]) # Not on factory tile
            & (board_sum > 0) # Is on resource tile
            & (lichen_allegiance_map != allegiance[ally]), # Not on ally lichen tile
            True,
            False
        )

        action_masks['self_destruct'] = np.where(
            factories_map != 1,
            True,
            False
        )

        for unit_id, unit in obs.units[agent].items():
            row,col = unit.pos
            if unit.power < unit.action_queue_cost() * 2:
                for robot_action in self.robot_actions:
                    if robot_action == 'pickup' and unit.on_factory_tile(obs):
                        for _,factory in obs.factories[agent].items():
                            if factory.strain_id == obs.board.factory_occupancy_map[row,col]:
                                action_masks[robot_action][row, col] = True
                                break
                        continue
                    if robot_action == 'recharge':
                        action_masks[robot_action][row, col] = True
                        continue
                    action_masks[robot_action][row, col] = False

            # Robots can only be on top of one another on allied city tiles where they also
            # shouldn't be blowing up, so we can keep this as a single action/mask instead of seperate
            # light and HEAVY actions/masks
            if unit.power < unit.self_destruct_cost():
                action_masks['self_destruct'][row, col] = False
            if unit.power < unit.dig_cost(obs):
                action_masks['dig'][row, col] = False

            for direction in move_directions.keys():
                if action_masks[f'move_{direction}'][row,col]:
                    if unit.power < unit.move_cost(obs, direction):
                        action_masks[f'move_{direction}'][row, col] = False

            # for direction in move_directions.keys():
            #     print(direction)
            #     print(action_masks[f'move_{direction}'].astype(int))
            # return

        return action_masks
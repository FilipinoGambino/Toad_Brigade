import sys
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gym import spaces

from abc import ABC

from lux_ai.lux.game_state import GameState

import luxai_s2.unit as luxai_unit
from luxai_s2.unit import UnitType
from luxai_s2.config import EnvConfig
from ..lux_gym.act_spaces import (
    FactoryBuildAction,
    FactoryWaterAction,
    MoveAction,
    PickupAction,
    DigAction,
    SelfDestructAction,
    RechargeAction,
)

MAP_SIZE = EnvConfig.map_size
ROBOT_ACTIONS = 8 + 1
ROBOTS = EnvConfig().ROBOTS
ACTIONS = [
    MoveAction(move_dir=0), # no-op
    MoveAction(move_dir=1), # move up
    MoveAction(move_dir=2), # move right
    MoveAction(move_dir=3), # move down
    MoveAction(move_dir=4), # move left
    PickupAction(resource=4), # pickup power to unit's capacity
    DigAction(),
    SelfDestructAction(),
    RechargeAction(), # 5 steps enough? Most of the energy should be coming from factories as recharging is too slow
    FactoryBuildAction(str(UnitType.LIGHT)),
    FactoryBuildAction(str(UnitType.HEAVY)),
    FactoryWaterAction(),
]

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

        self.directions = dict(center=0, up=1, right=2, down=3, left=4)
        self.resources = dict(ice=0, ore=1, water=2, metal=3, power=4)

        self.factory_actions = len(ACTIONS) - 3

        n_actions = len(ACTIONS)
        map_shape = (n_actions, MAP_SIZE, MAP_SIZE)
        action_space = spaces.Discrete(n_actions)

        super().__init__(action_space)

    def action_to_lux_action(self, agent: str, obs: GameState, actions: npt.NDArray):
        lux_action = dict()

        unit_actions = actions.copy()
        factory_actions = actions.copy()

        unit_actions[ROBOT_ACTIONS:] = -1e8
        factory_actions[:ROBOT_ACTIONS] = -1e8

        # print([layer for layer in factory_actions])
        # print(f"unit actions: {np.argwhere(unit_actions > -1000)}")
        # print(f"factory actions: {np.argwhere(factory_actions > -1000)}")

        unit_actions = unit_actions.argmax(axis=0)
        factory_actions = factory_actions.argmax(axis=0)
        print(unit_actions, factory_actions)

        for unit_id, unit in obs.units[agent].items():
            weight_class = unit.unit_type
            bot_cfg = ROBOTS[weight_class]
            row,col = unit.pos
            power_space = min(bot_cfg.BATTERY_CAPACITY - unit.power, obs.board.power[row,col])
            steps = 0

            action = unit_actions[row,col]

            if action != 0: # Not no_op
                action_queue = [
                    ACTIONS[action](
                        pickup_amount=power_space,
                        repeat=1,
                        n=steps,
                    )
                ]
                if unit.action_queue:
                    if unit.action_queue[0] != action_queue[0]: # FIXME if sequence of actions is used
                        lux_action[unit_id] = action_queue

        for factory_id, factory in obs.factories[agent].items():
            row, col = factory.pos
            action = factory_actions[row, col]
            lux_action[factory_id] = ACTIONS[action](repeat=0, n=0)
        print(lux_action)
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
        action_masks = {action: np.ones(
            map_shape, dtype=int
        )
            for action_space in self.flags['action_spaces'].keys()
            for action in self.flags['action_spaces'][action_space]
        }


        # Robot action masks
        move_mask = np.ones(  # 0: center, 1: up, 2: right, 3: down, 4: left
            (5, *map_shape), dtype=int
        )
        # transfer_mask = np.zeros( # power/ore/ice * directions (c/u/d/l/r) Maybe 5 dir layers and 3 res layers
        #     (15, *map_shape), dtype=bool
        # )
        # pickup_mask = np.zeros_like(
        #     (1, *map_shape), dtype=bool
        # )
        # dig_mask = np.zeros_like(
        #     (1, *map_shape), dtype=bool
        # )
        self_destruct_mask = np.ones_like(
            (1, *map_shape), dtype=bool
        )
        recharge_mask = np.ones_like(
            (1, *map_shape), dtype=bool
        )

        # Factory action masks
        build_light_mask = np.zeros(
            (1, *map_shape), dtype=int
        )
        build_heavy_mask = np.zeros(
            (1, *map_shape), dtype=int
        )
        grow_lichen_mask = np.ones( # Always true
            (1, *map_shape), dtype=int
        )


        factories_map = np.zeros(
            (1, *map_shape), dtype=int
        )


        # Out of bounds
        action_masks['move_up'][0, :] = False
        action_masks['move_right'][:, -1] = False
        action_masks['move_down'][-1, :] = False
        action_masks['move_left'][:, 0] = False

        ally_lichen_strains = {player.agent: player.factory_strains for player in obs.players}
        lichen_allegiance = np.zeros(map_shape)

        # 0: center, 1: up, 2: right, 3: down, 4: left
        deltas = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])

        for player in obs.players.keys():
            for factory_id, factory in obs.factories[player].items():
                row,col = factory.pos

                factories_map[ # Factories are always 3x3 so no need to check out-of-bounds
                    0,
                    row - 1: row + 2,
                    col - 1: col + 2,
                ] = (agent != player) + 1  # 0: no factory, 1: ally factory, 2: enemy factory

                if agent == player:
                    if factory.can_build_light():
                        action_masks['build_light'][row, col] = True
                    if factory.can_build_heavy():
                        action_masks['build_heavy'][row, col] = True

                    for strain in player.factory_strains:
                        lichen_allegiance = lichen_allegiance()

                else:
                    # Masks moving from (rol,col) position to illegal position
                    for i, (drow, dcol) in enumerate(deltas):
                        action_masks[f'move_{self.directions.keys()[i]}'][
                            max(0, row - 1 + drow): min(row + 2 + drow, MAP_SIZE),
                            max(0, col - 1 + dcol): min(col + 2 + dcol, MAP_SIZE),
                        ] = False



        # coords = np.argwhere(factories_map == 1)
        # pickup_mask.ravel()[np.ravel_multi_index(coords.T, pickup_mask.shape)] = True
        pickup_mask = np.where(
            factories_map == 1,
            1,
            0
        )

        _, index = np.unique(obs.board.lichen_strains, return_inverse=True)
        ally_lichen_map = ally_lichen_strains[index].reshape(map_shape)

        board_sum = obs.board.board_sum

        dig_mask = np.where(
            (factories_map == 0) # Not on factory tile
            & (board_sum > 0) # Is on resource tile
            & (ally_lichen_map == 0), # Not on ally lichen tile
            1,
            0
        )
        self_destruct_mask = np.where(
            factories_map != 1,
            1,
            0
        )

        for unit_id, unit in obs.units[agent].items():
            row,col = unit.pos

            # Robots can only be on top of one another on allied city tiles where they also
            # shouldn't be blowing up, so we can keep this as a single action/mask instead of seperate
            # light and HEAVY actions/masks
            if unit.power < unit.self_destruct_cost():
                self_destruct_mask[0, row, col] = False

            for idx,_ in enumerate(deltas):
                if unit.power < unit.move_cost(obs, idx):
                    move_mask[
                        idx,
                        row,
                        col
                    ] = False

        action_mask = np.concatenate((
            move_mask,
            # transfer_mask,
            dig_mask,
            pickup_mask,
            self_destruct_mask,
            recharge_mask,
            build_light_mask,
            build_heavy_mask,
            grow_lichen_mask,
        ), axis=0, dtype=int)

        return action_mask
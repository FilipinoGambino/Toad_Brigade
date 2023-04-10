"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys
import numpy as np
import torch
from typing import Any
from pathlib import Path
from types import SimpleNamespace
import yaml

from luxai_s2.env import LuxAI_S2
from lux_ai.lux.game_state import GameState

from stable_baselines3.ppo import PPO
from lux_ai.lux.config import EnvConfig
from lux_ai.lux_gym.controller import LuxController
# SimpleUnitObservationWrapper


# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"
RL_AGENT_CONFIG_PATH = Path(__file__).parent / "agent_config.yaml"


def my_turn_to_place_factory(place_first: bool, step: int):
    if place_first:
        if step % 2 == 0:
            return True
    else:
        if step % 2 == 1:
            return True
    return False


class Agent:
    def __init__(
            self,
            player: str,
            env_cfg: EnvConfig,
            controller: LuxController = None,
            policy: Any = None,
            flags: SimpleNamespace = None,
    ) -> None:
        self.agent_flags = flags

        self.my_id = player
        self.opp_player = "player_1" if self.my_id == "player_0" else "player_0"
        np.random.seed(42)
        self.env_cfg: EnvConfig = env_cfg
        self.factories_to_place = self.env_cfg.MAX_FACTORIES
        self.bidding_done = False
        self.controller = controller
        self.policy = policy
        self.flags = flags

        # directory = osp.dirname(__file__)
        # self.policy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))

        # self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def bid_policy(self, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        if self.my_id == 'player_0':
            return dict(faction="AlphaStrike", bid=0)
        else:
            return dict(faction="TheBuilders", bid=0)

    def factory_placement_policy(self, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        if obs.players[self.my_id].metal == 0:
            return dict()
        potential_spawns = list(zip(*np.where(obs.board.valid_spawns_mask == 1)))
        potential_spawns_set = set(potential_spawns)
        done_search = False

        ice_diff = np.diff(obs.board.ice)
        pot_ice_spots = np.argwhere(ice_diff == 1)
        if len(pot_ice_spots) == 0:
            pot_ice_spots = potential_spawns
        trials = 5
        while trials > 0:
            pos_idx = np.random.randint(0, len(pot_ice_spots))
            pos = pot_ice_spots[pos_idx]

            area = 3
            for x in range(area):
                for y in range(area):
                    check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                    if tuple(check_pos) in potential_spawns_set:
                        done_search = True
                        pos = check_pos
                        break
                if done_search:
                    break
            if done_search:
                break
            trials -= 1
        pos = potential_spawns[np.random.randint(0, len(potential_spawns))]
        water_metal = self.env_cfg.INIT_WATER_METAL_PER_FACTORY
        return dict(spawn=pos, metal=water_metal, water=water_metal)

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if not self.bidding_done:
            self.bidding_done = True
            return self.bid_policy
        else: # factory placement period
            # how many factories you have left to place
            factories_to_place = self.factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(obs.players[self.my_id].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                return self.factory_placement_policy(step,obs)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        # raw_obs = dict(player_0=obs, player_1=obs)
        # obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        with torch.no_grad():
            # to improve performance, we have a rule based action mask generator for the controller used
            # which will force the agent to generate actions that are valid only.
            action_mask = self.controller.action_masks(self.my_id, obs)
            # for key, mask in action_mask.items():
            #     print([f"{key}:  {mask}"])

            # SB3 doesn't support invalid action masking. So we do it ourselves here
            # logits = self.policy(obs.unsqueeze(0)) # FIXME Start the policy!!!
            #
            # logits[~action_mask] = -1e8  # mask out invalid actions
            # dist = torch.distributions.Categorical(logits=logits)
            # actions = dist.sample().cpu().numpy()  # shape (1, 1)
            action_flag = self.flags.actions
            map_flag = self.flags.map_shape

            action_maps = {
                action: np.random.uniform(low=0, high=1, size=map_flag)
                for unit,actions in action_flag.items()
                for action in actions
            }
            for unit,actions in action_flag.items():
                for action in actions:
                    mask = action_mask[action]
                    action_maps[action][~mask] = -1e8

        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        lux_action = self.controller.action_to_lux_action(
            self.my_id,
            obs,
            action_maps
        )

        return lux_action

    @property
    def unwrapped_env(self):# -> LuxEnv:
        return self.env.unwrapped[0]

    @property
    def game_state(self) -> GameState:
        return self.unwrapped_env.game_state
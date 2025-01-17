"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys
import numpy as np
import torch
from typing import Any, Dict
from pathlib import Path
from types import SimpleNamespace
import yaml

from luxai_s2.env import LuxAI_S2
from lux_ai.lux.game_state import GameState
from ..nns import create_model

from stable_baselines3.ppo import PPO
from lux_ai.lux.config import EnvConfig
from lux_ai.lux_gym.controller import LuxController

# SimpleUnitObservationWrapper


# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"
RL_AGENT_CONFIG_PATH = Path(__file__).parent / "agent_config.yaml"
MODEL_CONFIG_PATH = Path(__file__).parent / "model_config.yaml"


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
            policy: Any = None,
    ) -> None:
        with open(RL_AGENT_CONFIG_PATH, 'r') as f:
            self.agent_flags = SimpleNamespace(**yaml.full_load(f))
        with open(MODEL_CONFIG_PATH, 'r') as f:
            self.model_flags = SimpleNamespace(**yaml.full_load(f))

        self.my_id = player
        self.opp_id = "player_1" if self.my_id == "player_0" else "player_0"
        int_id = int(self.my_id[-1])

        if torch.cuda.is_available():
            device_id = f"cuda:{min(int_id, torch.cuda.device_count() - 1)}"
        else:
            device_id = "cpu"
        self.device = torch.device(device_id)

        np.random.seed(42)
        self.env_cfg: EnvConfig = env_cfg
        self.factories_to_place = self.env_cfg.MAX_FACTORIES
        self.bidding_done = False

        self.controller = LuxController(self.env_cfg, self.agent_flags)
        self.policy = policy
        self.model = create_model(self.model_flags, self.device)
        print(self.model)

        # directory = osp.dirname(__file__)
        # self.policy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))


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

            # SB3 doesn't support invalid action masking. So we do it ourselves here
            # logits = self.policy(obs.unsqueeze(0)) # FIXME Start the policy!!!
            #
            # logits[~action_mask] = -1e8  # mask out invalid actions
            # dist = torch.distributions.Categorical(logits=logits)
            # actions = dist.sample().cpu().numpy()  # shape (1, 1)
            action_flag = self.agent_flags.actions
            map_flag = self.agent_flags.map_shape

            action_maps = {
                action: np.random.uniform(low=0, high=1, size=map_flag)
                for unit,actions in action_flag.items()
                for action in actions
            }
            for actions in action_flag.values():
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

    def preprocess(self, obs: GameState) -> Dict[str, torch.tensor]:
        tensors = {
            'maps': dict(
                ice=obs.board.ice, # Boolean
                ore=obs.board.ore, # Boolean
                robots=obs.board.robots,  # Boolean
                lichen_spreading=obs.board.lichen_spreading,  # Boolean
                cargo_light_full=obs.board.cargo_light_full,  # Boolean
                cargo_heavy_full=obs.board.cargo_heavy_full,  # Boolean
                robot_weight=obs.board.robot_weight,  # Discrete
                lichen_strains=obs.board.lichen_strains, # Discrete
                allegiance=obs.board.allegiance, # Discrete
                factory_strains=obs.board.factory_strains, # Discrete - Max factories * 2

                lichen=obs.board.lichen // self.env_cfg.MAX_LICHEN_PER_TILE,  # Cont
                power_factory=obs.board.power_factory,  # Cont
                rubble=obs.board.rubble // self.env_cfg.MAX_RUBBLE,  # Cont
                cargo_factory_ice=obs.board.cargo_factory_ice, # Cont - inf
                cargo_factory_ore=obs.board.cargo_factory_ore, # Cont - inf
                cargo_factory_water=obs.board.cargo_factory_water, # Cont - inf
                cargo_factory_metal=obs.board.cargo_factory_metal, # Cont - inf
                power_light=obs.board.power_light // self.env_cfg.ROBOTS['LIGHT'].BATTERY_CAPACITY, # Cont
                power_heavy=obs.board.power_heavy // self.env_cfg.ROBOTS['HEAVY'].BATTERY_CAPACITY, # Cont
                cargo_light_ice=obs.board.cargo_light_ice // self.env_cfg.ROBOTS['LIGHT'].CARGO_SPACE, # Cont
                cargo_light_ore=obs.board.cargo_light_ore // self.env_cfg.ROBOTS['LIGHT'].CARGO_SPACE, # Cont
                cargo_heavy_ice=obs.board.cargo_heavy_ice // self.env_cfg.ROBOTS['HEAVY'].CARGO_SPACE, # Cont
                cargo_heavy_ore=obs.board.cargo_heavy_ore // self.env_cfg.ROBOTS['HEAVY'].CARGO_SPACE, # Cont
        ),
            'globals': dict(
                my_lichen=obs.players[self.my_id].lichen_count, # Boolean
                opp_lichen=obs.players[self.opp_id].lichen_count, # Boolean
                env_steps=obs.real_env_steps, # Cont
                game_phase=obs.game_phase(), # Discrete
                cycle_step=obs.cycle_step(), # Discrete
                is_day=obs.is_day(), # Boolean
            )
        }
        for key in tensors['maps'].keys():
            tensors[key] = torch.from_numpy(tensors[key])

        return tensors
    #
    #
    #
    # @property
    # def unwrapped_env(self):# -> LuxEnv:
    #     return self.env.unwrapped[0]
    #
    # @property
    # def game_state(self) -> GameState:
    #     return self.unwrapped_env.game_state
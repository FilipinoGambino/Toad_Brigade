import copy
import gym
from gym import spaces
import numpy as np
import torch
from typing import Dict, List, NoReturn, Optional, Tuple, Union, Any
import numpy.typing as npt
import itertools

from ..lux.config import EnvConfig

#
# # from .act_spaces import ACTION_MEANINGS
# from .lux_env import LuxEnv
# from .reward_spaces import BaseRewardSpace
# from ..utility_constants import MAP_SIZE
#
#
# class SharedObs(gym.Wrapper):
#     @staticmethod
#     def _get_shared_observation(obs: Dict[str, np.array]) -> np.array:
#         assert obs['player_0'] == obs['player_1'], "player observations not identical"
#         shared_obs = obs['player_0']
#         return shared_obs
#
#     def reset(self, **kwargs):
#         return SharedObs._get_shared_observation(super(SharedObs, self).reset(**kwargs))
#
#     def step(self, actions):
#         return SharedObs._get_shared_observation(super(SharedObs, self).step(actions))
#
#
# class RewardSpaceWrapper(gym.Wrapper):
#     def __init__(self, env: LuxEnv, reward_space: BaseRewardSpace):
#         super(RewardSpaceWrapper, self).__init__(env)
#         self.reward_space = reward_space
#
#     def _get_rewards_and_done(self) -> Tuple[Tuple[float, float], bool]:
#         rewards, done = self.reward_space.compute_rewards_and_done(self.unwrapped.game_state, self.done)
#         if self.unwrapped.done and not done:
#             raise RuntimeError("Reward space did not return done, but the lux engine is done.")
#         self.unwrapped.done = done
#         return rewards, done
#
#     def reset(self, **kwargs):
#         obs, _, _, info = super(RewardSpaceWrapper, self).reset(**kwargs)
#         return (obs, *self._get_rewards_and_done(), info)
#
#     def step(self, action):
#         obs, _, _, info = super(RewardSpaceWrapper, self).step(action)
#         return (obs, *self._get_rewards_and_done(), info)
#
#
# # TODO Add logging info
# class LoggingEnv(gym.Wrapper):
#     def __init__(self, env: gym.Env, reward_space: BaseRewardSpace):
#         super(LoggingEnv, self).__init__(env)
#         self.reward_space = reward_space
#         self.vals_peak = {}
#         self.reward_sums = [0., 0.]
#         # self.actions_distributions = {
#         #     f"{space}.{meaning}": 0.
#         #     for space, action_meanings in ACTION_MEANINGS.items()
#         #     for meaning in action_meanings
#         # }
#         # TODO: Resource mining % like in visualizer?
#         # self.resource_count = {"wood", etc...}
#         # TODO: Fuel metric?
#
#     def info(self, info: Dict[str, np.ndarray], rewards: List[int]) -> Dict[str, np.ndarray]:
#         info = copy.copy(info)
#         game_state = self.env.unwrapped.game_state
#         logs = {
#             "step": [game_state.turn],
#             "city_tiles": [p.city_tile_count for p in game_state.players],
#             "separate_cities": [len(p.cities) for p in game_state.players],
#             "workers": [sum(u.is_worker() for u in p.units) for p in game_state.players],
#             "carts": [sum(u.is_cart() for u in p.units) for p in game_state.players],
#             "research_points": [p.research_points for p in game_state.players],
#         }
#         self.vals_peak = {
#             key: np.maximum(val, logs[key]) for key, val in self.vals_peak.items()
#         }
#         for key, val in self.vals_peak.items():
#             logs[f"{key}_peak"] = val.copy()
#             logs[f"{key}_final"] = logs[key]
#             del logs[key]
#
#         self.reward_sums = [r + s for r, s in zip(rewards, self.reward_sums)]
#         logs["mean_cumulative_rewards"] = [np.mean(self.reward_sums)]
#         logs["mean_cumulative_reward_magnitudes"] = [np.mean(np.abs(self.reward_sums))]
#         logs["max_cumulative_rewards"] = [np.max(self.reward_sums)]
#
#         actions_taken = self.env.unwrapped.action_space.actions_taken_to_distributions(info["actions_taken"])
#         self.actions_distributions = {
#             f"{space}.{act}": self.actions_distributions[f"{space}.{act}"] + n
#             for space, dist in actions_taken.items()
#             for act, n in dist.items()
#         }
#         logs.update({f"ACTIONS_{key}": val for key, val in self.actions_distributions.items()})
#
#         info.update({f"LOGGING_{key}": np.array(val, dtype=np.float32) for key, val in logs.items()})
#         # Add any additional info from the reward space
#         info.update(self.reward_space.get_info())
#         return info
#
#     def reset(self, **kwargs):
#         obs, reward, done, info = super(LoggingEnv, self).reset(**kwargs)
#         self._reset_peak_vals()
#         self.reward_sums = [0., 0.]
#         self.actions_distributions = {
#             key: 0. for key in self.actions_distributions.keys()
#         }
#         return obs, reward, done, self.info(info, reward)
#
#     def step(self, action: Dict[str, np.ndarray]):
#         obs, reward, done, info = super(LoggingEnv, self).step(action)
#         return obs, reward, done, self.info(info, reward)
#
#     def _reset_peak_vals(self) -> NoReturn:
#         self.vals_peak = {
#             key: np.array([0., 0.])
#             for key in [
#                 "city_tiles",
#                 "separate_cities",
#                 "workers",
#                 "carts",
#             ]
#         }
#
#
# class VecEnv(gym.Env):
#     def __init__(self, envs: List[gym.Env]):
#         self.envs = envs
#         self.last_outs = [() for _ in range(len(self.envs))]
#
#     @staticmethod
#     def _stack_dict(x: List[Union[Dict, np.ndarray]]) -> Union[Dict, np.ndarray]:
#         if isinstance(x[0], dict):
#             return {key: VecEnv._stack_dict([i[key] for i in x]) for key in x[0].keys()}
#         else:
#             return np.stack([arr for arr in x], axis=0)
#
#     @staticmethod
#     def _vectorize_env_outs(env_outs: List[Tuple]) -> Tuple:
#         obs_list, reward_list, done_list, info_list = zip(*env_outs)
#         obs_stacked = VecEnv._stack_dict(obs_list)
#         reward_stacked = np.array(reward_list)
#         done_stacked = np.array(done_list)
#         info_stacked = VecEnv._stack_dict(info_list)
#         return obs_stacked, reward_stacked, done_stacked, info_stacked
#
#     def reset(self, force: bool = False, **kwargs):
#         if force:
#             # noinspection PyArgumentList
#             self.last_outs = [env.reset(**kwargs) for env in self.envs]
#             return VecEnv._vectorize_env_outs(self.last_outs)
#
#         for i, env in enumerate(self.envs):
#             # Check if env finished
#             if self.last_outs[i][2]:
#                 # noinspection PyArgumentList
#                 self.last_outs[i] = env.reset(**kwargs)
#         return VecEnv._vectorize_env_outs(self.last_outs)
#
#     def step(self, action: Dict[str, np.ndarray]):
#         actions = [
#             {key: val[i] for key, val in action.items()} for i in range(len(self.envs))
#         ]
#         self.last_outs = [env.step(a) for env, a in zip(self.envs, actions)]
#         return VecEnv._vectorize_env_outs(self.last_outs)
#
#     def render(self, mode: str = "human", **kwargs):
#         # noinspection PyArgumentList
#         return self.envs[kwargs["idx"]].render(mode, **kwargs)
#
#     def close(self):
#         return [env.close() for env in self.envs]
#
#     def seed(self, seed: Optional[int] = None) -> list:
#         if seed is not None:
#             return [env.seed(seed + i) for i, env in enumerate(self.envs)]
#         else:
#             return [env.seed(seed) for i, env in enumerate(self.envs)]
#
#     @property
#     def unwrapped(self) -> List[gym.Env]:
#         return [env.unwrapped for env in self.envs]
#
#     @property
#     def action_space(self) -> List[gym.spaces.Dict]:
#         return [env.action_space for env in self.envs]
#
#     @property
#     def observation_space(self) -> List[gym.spaces.Dict]:
#         return [env.observation_space for env in self.envs]
#
#     @property
#     def metadata(self) -> List[Dict]:
#         return [env.metadata for env in self.envs]
#
#
# class PytorchEnv(gym.Wrapper):
#     def __init__(self, env: Union[gym.Env, VecEnv], device: torch.device = torch.device("cpu")):
#         super(PytorchEnv, self).__init__(env)
#         self.device = device
#
#     def reset(self, **kwargs):
#         return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).reset(**kwargs)])
#
#     def step(self, action: Dict[str, torch.Tensor]):
#         action = {
#             key: val.cpu().numpy() for key, val in action.items()
#         }
#         return tuple([self._to_tensor(out) for out in super(PytorchEnv, self).step(action)])
#
#     def _to_tensor(self, x: Union[Dict, np.ndarray]) -> Dict[str, Union[Dict, torch.Tensor]]:
#         if isinstance(x, dict):
#             return {key: self._to_tensor(val) for key, val in x.items()}
#         else:
#             return torch.from_numpy(x).to(self.device, non_blocking=True)
#
#
# class DictEnv(gym.Wrapper):
#     @staticmethod
#     def _dict_env_out(env_out: tuple) -> dict:
#         obs, reward, done, info = env_out
#         assert "obs" not in info.keys()
#         assert "reward" not in info.keys()
#         assert "done" not in info.keys()
#         return dict(
#             obs=obs,
#             reward=reward,
#             done=done,
#             info=info
#         )
#
#     def reset(self, **kwargs):
#         return DictEnv._dict_env_out(super(DictEnv, self).reset(**kwargs))
#
#     def step(self, action):
#         return DictEnv._dict_env_out(super(DictEnv, self).step(action))

class ObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController
    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.
    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory
    """

    def __init__(self, env_cfg: EnvConfig, sample_obs) -> None:
        super().__init__(sample_obs)
        p = 2  # number of players
        env_cfg = env_cfg
        x = y = env_cfg.map_size
        self.observation_space = spaces.Dict({
            # none, robot
            "light_robot": spaces.MultiBinary((1, p, x, y)),
            "heavy_robot": spaces.MultiBinary((1, p, x, y)),

            # LIGHT ROBOTS
            # Number of units in the square (only relevant on city tiles)
            "light_count": spaces.Box(0., float("inf"), shape=(1, p, x, y)),
            # light robot power normalized from 0-150
            "light_power": spaces.Box(0., 1., shape=(1, p, x, y)),
            # light robot cargo normalized from 0-100
            "light_ice": spaces.Box(0., 1., shape=(1, p, x, y)),
            "light_ore": spaces.Box(0., 1., shape=(1, p, x, y)),
            "light_water": spaces.Box(0., 1., shape=(1, p, x, y)),
            "light_metal": spaces.Box(0., 1., shape=(1, p, x, y)),
            # normalized to light robot capacity
            "light_full": spaces.MultiBinary((1, p, x, y)),

            # HEAVY ROBOTS
            # Number of units in the square (only relevant on city tiles)
            "heavy_count": spaces.Box(0., float("inf"), shape=(1, p, x, y)),
            # heavy robot power normalized from 0-3000
            "heavy_power": spaces.Box(0., 1., shape=(1, p, x, y)),
            # heavy robot cargo normalized from 0-1000
            "heavy_ice": spaces.Box(0., 1., shape=(1, p, x, y)),
            "heavy_ore": spaces.Box(0., 1., shape=(1, p, x, y)),
            "heavy_water": spaces.Box(0., 1., shape=(1, p, x, y)),
            "heavy_metal": spaces.Box(0., 1., shape=(1, p, x, y)),
            # robot cargo full
            "heavy_full": spaces.MultiBinary((1, p, x, y)),

            # FACTORIES
            "factory": spaces.MultiBinary((1, p, x, y)),
            "factory_power": spaces.Box(0., 1., shape=(1, p, x, y)),
            "factory_ice": spaces.Box(0., 1., shape=(1, p, x, y)),
            "factory_ore": spaces.Box(0., 1., shape=(1, p, x, y)),
            "factory_water": spaces.Box(0., 1., shape=(1, p, x, y)),
            "factory_metal": spaces.Box(0., 1., shape=(1, p, x, y)),
            "factory_strain": spaces.MultiDiscrete(np.full((1, p, x, y), env_cfg.MAX_FACTORIES)),

            # TODO What was this for again?
            # "repeats_remaining": spaces.MultiDiscrete(np.full((1,1), 50)),

            # MAPS
            # lichen count greater than spreading minimum and less than max
            "lichen_spreading": spaces.MultiBinary((1, 1, x, y)),
            # lichen strain ID
            "lichen_strain": spaces.MultiDiscrete(np.full((1, 1, x, y), env_cfg.MAX_FACTORIES)),
            # Resources
            "lichen": spaces.Box(0., 1., shape=(1, 1, x, y)),
            "rubble": spaces.Box(0., 1., shape=(1, 1, x, y)),
            "ore": spaces.Box(0., 1., shape=(1, 1, x, y)),
            "ice": spaces.Box(0., 1., shape=(1, 1, x, y)),

            # GLOBALS
            # turn number // 100
            "game_phase": spaces.MultiDiscrete(np.full((1, 1), 10)),
            # turn number % 50
            "cycle_step": spaces.MultiDiscrete(np.full((1, 1), env_cfg.CYCLE_LENGTH)),
            # turn number, normalized from 0-1000
            "turn": spaces.Box(0., 1., shape=(1, 1)),
            # true during the day
            "is_day": spaces.MultiDiscrete(np.full((1, 1), 2)),
        })
        self._empty_obs = {}

    def observation(self, observation):
        for key, spec in self.observation_space.spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")
        return ObservationWrapper.convert_obs(observation, self.env.state.env_cfg, self._empty_obs)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(observation: Dict[str, Any], env_cfg: Any, empty_obs: Dict[str, Any]) -> Dict[str, npt.NDArray]:
        obs = empty_obs
        shared_obs = observation['player_0']
        board_maps = shared_obs["board"]

        for p_idx, p_id in enumerate(observation.keys()):
            for u_id,unit in shared_obs['units'][p_id].items():
                cargo_space = env_cfg.ROBOTS[unit['unit_type']].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit['unit_type']].BATTERY_CAPACITY
                weight_class = unit['unit_type'].lower()
                x, y = unit['pos']

                obs[f'{weight_class}_robot'][0, p_idx, x, y] = 1
                obs[f'{weight_class}_count'][0, p_idx, x, y] += 1
                obs[f'{weight_class}_power'][0, p_idx, x, y] = unit['power'] / battery_cap
                obs[f'{weight_class}_ice'][0, p_idx, x, y] = unit['cargo']['ice'] / cargo_space
                obs[f'{weight_class}_ore'][0, p_idx, x, y] = unit['cargo']['ore'] / cargo_space
                obs[f'{weight_class}_water'][0, p_idx, x, y] = unit['cargo']['water'] / cargo_space
                obs[f'{weight_class}_metal'][0, p_idx, x, y] = unit['cargo']['metal'] / cargo_space
                obs[f'{weight_class}_cargo_full'][0, p_idx, x, y] = sum(unit['cargo'].values()) == cargo_space

            factories = shared_obs['factories'][p_id]
            for f_id,factory in factories.items():
                # Lichen also gives power, but I'm hoping robots are picking up power to keep the available power lower
                # Box space will keep it limited to 1.0
                power_cap = env_cfg.INIT_POWER_PER_FACTORY + env_cfg.max_episode_length * env_cfg.FACTORY_CHARGE
                x, y = factory['pos']

                obs['factory'][0, p_idx, x-1 : x+2, y-1 : y+2] = 1
                obs['factory_power'][0, p_idx, x-1 : x+2, y-1 : y+2] = factory['power'] / power_cap
                obs['factory_ice'][0, p_idx, x-1 : x+2, y-1 : y+2] = factory['cargo']['ice']
                obs['factory_ore'][0, p_idx, x-1 : x+2, y-1 : y+2] = factory['cargo']['ore']
                obs['factory_water'][0, p_idx, x-1 : x+2, y-1 : y+2] = factory['cargo']['water']
                obs['factory_metal'][0, p_idx, x-1 : x+2, y-1 : y+2] = factory['cargo']['metal']
                obs['factory_strain'][0, p_idx, x-1 : x+2, y-1 : y+2] = factory['strain_id']

            for x,y in itertools.product(range(env_cfg.map_size), repeat=2):
                obs['rubble'][0, 0, x, y] = board_maps["rubble"][x, y] / 5000
                obs['ore'][0, 0, x, y] = board_maps["ore"][x, y] / 5000
                obs['ice'][0, 0, x, y] = board_maps["ice"][x, y] / 5000
                obs['lichen'][0, 0, x, y] = board_maps["rubble"][x, y] / 100
                obs['lichen_spreading'][0, 0, x, y] = env_cfg.MIN_LICHEN_TO_SPREAD <= board_maps['lichen'][x, y] < env_cfg.MAX_LICHEN_PER_TILE
                obs['lichen_strain'][0, 0, x, y] = board_maps["lichen_strains"][x, y]

            obs['game_phase'][0, 0] = shared_obs['real_env_steps'] // 100
            obs['cycle_step'][0, 0] = shared_obs['real_env_steps'] % env_cfg.CYCLE_LENGTH
            obs['turn'][0, 0] = shared_obs['real_env_steps'] / env_cfg.max_episode_length
            obs['is_day'][0, 0] = shared_obs['real_env_steps'] % env_cfg.CYCLE_LENGTH < env_cfg.DAY_LENGTH

        return obs
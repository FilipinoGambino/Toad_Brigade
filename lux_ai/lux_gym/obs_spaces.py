import functools
import logging
from abc import ABC, abstractmethod
import gym
import itertools
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Any

from lux_ai.lux.game_state import GameState
from ..utility_constants import MAP_SIZE, MAX_FACTORIES, CYCLE_LENGTH

# Player count
P = 2


class BaseObsSpace(ABC):
    # NB: Avoid using Discrete() space, as it returns a shape of ()
    # NB: "_COUNT" keys indicate that the value is used to scale the embedding of another value
    @abstractmethod
    def get_obs_spec(self) -> gym.spaces.Dict:
        pass

    @abstractmethod
    def wrap_env(self, env) -> gym.Wrapper:
        pass


class FixedShapeObs(BaseObsSpace, ABC):
    def get_obs_spec(self) -> gym.spaces.Dict:
        x,y = MAP_SIZE
        return gym.spaces.Dict({
            # robot presence by weight class
            "light_robot": gym.spaces.MultiBinary((1, P, x, y)),
            "heavy_robot": gym.spaces.MultiBinary((1, P, x, y)),

            ## LIGHT ROBOTS
            # Number of units in the square (only relevant on city tiles)
            "light_count": gym.spaces.Box(0., float("inf"), shape=(1, P, x, y)),
            # light robot power normalized from 0-150
            "light_power": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # light robot cargo normalized from 0-100
            "light_ice": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "light_ore": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "light_water": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "light_metal": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # normalized to light robot capacity
            "light_full": gym.spaces.MultiBinary((1, P, x, y)),

            ## HEAVY ROBOTS
            # Number of units in the square (only relevant on city tiles)
            "heavy_count": gym.spaces.Box(0., float("inf"), shape=(1, P, x, y)),
            # heavy robot power normalized from 0-3000
            "heavy_power": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # heavy robot cargo normalized from 0-1000
            "heavy_ice": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "heavy_ore": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "heavy_water": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "heavy_metal": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            # robot cargo full
            "heavy_full": gym.spaces.MultiBinary((1, P, x, y)),

            ## FACTORIES
            "factory": gym.spaces.MultiBinary((1, P, x, y)),
            "factory_power": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "factory_ice": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "factory_ore": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "factory_water": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "factory_metal": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "factory_strain": gym.spaces.MultiDiscrete(np.full((1, P, x, y), MAX_FACTORIES * 2)),

            ## MAPS
            # lichen count greater than spreading minimum and less than max
            "lichen_spreading": gym.spaces.MultiBinary((1, 1, x, y)),
            # lichen strain ID
            "lichen_strain": gym.spaces.MultiDiscrete(np.full((1, 1, x, y), MAX_FACTORIES * 2)),
            # lichen count per tile normalized from 0-100
            "lichen": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            # rubble count per tile normalized from 0-100
            "rubble": gym.spaces.Box(0., 1., shape=(1, 1, x, y)),
            # ore on tile
            "ore": gym.spaces.MultiBinary((1, 1, x, y)),
            # ice on tile
            "ice": gym.spaces.MultiBinary((1, 1, x, y)),

            ## GLOBALS
            # turn number // 100
            "game_phase": gym.spaces.MultiDiscrete(np.full((1, 1), 10)),
            # turn number % 50
            "cycle_step": gym.spaces.MultiDiscrete(np.full((1, 1), CYCLE_LENGTH)),
            # turn number, normalized from 0-1000
            "turn": gym.spaces.Box(0., 1., shape=(1, 1)),
            # true during the day
            "is_day": gym.spaces.MultiDiscrete(np.full((1, 1), 2)),
        })

    def wrap_env(self, env) -> gym.Wrapper:
        return _FixedShapeContinuousObsWrapper(env)


class _FixedShapeContinuousObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(_FixedShapeContinuousObsWrapper, self).__init__(env)
        self._empty_obs = {}
        for key, spec in FixedShapeObs().get_obs_spec().spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        for key, spec in self.observation_space.spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")
        return _FixedShapeContinuousObsWrapper.convert_obs(observation, self.env.state.env_cfg, self._empty_obs)

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
                coords = (0, p_idx, x, y)

                obs[f'{weight_class}_robot'][coords] = 1
                obs[f'{weight_class}_count'][coords] += 1
                obs[f'{weight_class}_power'][coords] = unit['power'] / battery_cap
                obs[f'{weight_class}_ice'][coords] = unit['cargo']['ice'] / cargo_space
                obs[f'{weight_class}_ore'][coords] = unit['cargo']['ore'] / cargo_space
                obs[f'{weight_class}_water'][coords] = unit['cargo']['water'] / cargo_space
                obs[f'{weight_class}_metal'][coords] = unit['cargo']['metal'] / cargo_space
                obs[f'{weight_class}_cargo_full'][coords] = sum(unit['cargo'].values()) == cargo_space

            factories = shared_obs['factories'][p_id]
            for f_id,factory in factories.items():
                # Lichen also gives power, but I'm hoping robots are picking up power to keep the available power lower
                # Box space will keep it limited to 1.0 after normalization
                power_cap = env_cfg.INIT_POWER_PER_FACTORY + env_cfg.max_episode_length * env_cfg.FACTORY_CHARGE
                x, y = factory['pos']
                square = (0, p_idx, slice(x-1, x+2), slice(y-1, y+2))

                obs['factory'][square] = 1
                obs['factory_power'][square] = factory['power'] / power_cap
                obs['factory_ice'][square] = factory['cargo']['ice']
                obs['factory_ore'][square] = factory['cargo']['ore']
                obs['factory_water'][square] = factory['cargo']['water']
                obs['factory_metal'][square] = factory['cargo']['metal']
                obs['factory_strain'][square] = factory['strain_id']

            for x,y in itertools.product(range(env_cfg.map_size), repeat=2):
                obs['rubble'][0, 0, x, y] = board_maps["rubble"][x, y] / env_cfg.MAX_RUBBLE
                obs['lichen'][0, 0, x, y] = board_maps["lichen"][x, y] / 100
                obs['lichen_spreading'][0, 0, x, y] = env_cfg.MIN_LICHEN_TO_SPREAD \
                                                      <= board_maps['lichen'][x, y] \
                                                      < env_cfg.MAX_LICHEN_PER_TILE
                obs['lichen_strain'][0, 0, x, y] = board_maps["lichen_strains"][x, y]

            obs['ore'] = board_maps['ore']
            obs['ice'] = board_maps['ice']
            obs['game_phase'][0, 0] = shared_obs['real_env_steps'] // 100
            obs['cycle_step'][0, 0] = shared_obs['real_env_steps'] % env_cfg.CYCLE_LENGTH
            obs['turn'][0, 0] = shared_obs['real_env_steps'] / env_cfg.max_episode_length
            obs['is_day'][0, 0] = shared_obs['real_env_steps'] % env_cfg.CYCLE_LENGTH < env_cfg.DAY_LENGTH
        return obs

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_dist_from_center_x(map_height: int, map_width: int) -> np.ndarray:
        pos = np.linspace(0, 2, map_width, dtype=np.float32)[None, :].repeat(map_height, axis=0)
        return np.abs(1 - pos)[None, None, :, :]

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_dist_from_center_y(map_height: int, map_width: int) -> np.ndarray:
        pos = np.linspace(0, 2, map_height)[:, None].repeat(map_width, axis=1)
        return np.abs(1 - pos)[None, None, :, :]

class MultiObs(BaseObsSpace):
    def __init__(self, named_obs_spaces: Dict[str, BaseObsSpace], *args, **kwargs):
        super(MultiObs, self).__init__(*args, **kwargs)
        self.named_obs_spaces = named_obs_spaces

    def get_obs_spec(
            self,
            board_dims: Tuple[int, int] = MAP_SIZE
    ) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            name + key: val
            for name, obs_space in self.named_obs_spaces.items()
            for key, val in obs_space.get_obs_spec(board_dims).spaces.items()
        })

    def wrap_env(self, env) -> gym.Wrapper:
        return _MultiObsWrapper(env, self.named_obs_spaces)

class _MultiObsWrapper(gym.Wrapper):
    def __init__(self, env, named_obs_spaces: Dict[str, BaseObsSpace]):
        super(_MultiObsWrapper, self).__init__(env)
        self.named_obs_space_wrappers = {key: val.wrap_env(env) for key, val in named_obs_spaces.items()}

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: GameState) -> Dict[str, np.ndarray]:
        return {
            name + key: val
            for name, obs_space in self.named_obs_space_wrappers.items()
            for key, val in obs_space.observation(observation).items()
        }
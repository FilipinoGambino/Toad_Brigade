from lux_ai.rl_agent.agent import Agent
from lux_ai.lux.config import EnvConfig
from lux_ai.lux.game_state import obs_to_game_state#, process_action
from lux_ai.lux_gym import wrappers

from luxai_s2.env import LuxAI_S2
from lux_ai.lux_gym.controller import LuxController

import yaml
from pathlib import Path
from types import SimpleNamespace

RL_AGENT_CONFIG_PATH = Path(__file__).parent / "agent_config.yaml"

### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict()  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()

def agent_fn(observation, configurations):
    action_queue = dict()
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step

    player = observation.my_id
    print(observation)
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        env_cfg = EnvConfig.from_dict(configurations["env_cfg"])

        agent_prev_obs[player] = dict()
        agent = agent_dict[player]
    agent = agent_dict[player]
    obs = obs_to_game_state(step, configurations['env_cfg'], observation)
    agent.step = step
    if obs.real_env_steps < 0:
        actions = agent.early_setup(step, obs, remainingOverageTime)
    else:
        actions = agent.act(step, obs, remainingOverageTime)
    action_queue[player] = to_json(actions)
    return action_queue

def to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(s) for s in obj]
    elif isinstance(obj, dict):
        out = {}
        for k in obj:
            out[k] = to_json(obj[k])
        return out
    else:
        return obj

def animate(imgs, _return=True, fps=10, filename="__temp__.mp4"):
    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(imgs, fps=fps)
    clip.write_videofile(filename, verbose=False, logger=None)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import sys

    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)
    np.set_printoptions(linewidth=sys.maxsize)
    np.set_printoptions(threshold=sys.maxsize)

    with open(RL_AGENT_CONFIG_PATH, 'r') as f:
        agent_flags = SimpleNamespace(**yaml.full_load(f))

    env = LuxAI_S2(collect_stats=True, verbose=4)
    env.env_cfg = env.state.env_cfg
    env.env_steps = env.state.env_steps
    agents = {
            player_id: Agent(
                player_id,
                env.env_cfg,
                controller=LuxController(env.env_cfg, agent_flags),
                policy=None,
                flags=agent_flags,
            )
            for player_id in env.possible_agents
        }

    env = wrappers.GameStateWrapper(env)
    env = wrappers.SinglePhaseWrapper(env, agents)
    # env = wrappers.ObservationWrapper(env)
    # env = wrappers.PytorchEnv(env)
    # TODO maybe CustromEnvWrapper fixes this?
    #  https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
    obs = env.reset(seed=42)

    img = env.render("rgb_array", width=640, height=640)

    steps = 1000
    imgs = [img]
    done = False
    while not done:
        if env.state.real_env_steps >= steps: break
        actions = {}
        for player in env.agents:
            step = env.state.real_env_steps

            player_actions = env.agents[player].act(step, obs)
            actions[player] = player_actions
        obs, rewards, dones, infos = env.step(actions)
        imgs += [env.render("rgb_array", width=640, height=640)]
        done = dones["player_0"] and dones["player_1"]
    animate(imgs)
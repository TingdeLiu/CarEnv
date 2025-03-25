from typing import Dict, Optional

import numpy as np
import pettingzoo
from pettingzoo.utils.env import AgentID, ObsType
import gymnasium
from gymnasium.utils import seeding

from .Agent import Agent
from .Problems.MultiAgentProblem import MultiAgentProblem
from .ReadOnlyCollections import ReadOnlyDict, ReadOnlyList
from .Object import Object
from .TireSlipObject import TireSlipObject
from .Metrics import Metric
from .Rendering.EnvRenderer import make_env_renderer


def _make_problem(env, config) -> MultiAgentProblem:
    from .Problems.MultiAgentIntersectionProblem import MultiAgentIntersectionProblem

    params = config['problem']

    known_types = {
        'marl_intersection': MultiAgentIntersectionProblem,
    }

    kwargs = {k: v for k, v in params.items() if k != 'type'}
    return known_types[params['type']](env, **kwargs)


class MultiAgentCarEnv(pettingzoo.ParallelEnv):
    metadata = {
        'render_modes': ["human", "rgb_array"]
    }

    def __init__(self, config=None, render_mode=None, render_kwargs=None):
        super(MultiAgentCarEnv).__init__()
        self._np_random: Optional[np.random.Generator] = None

        self._config = config

        self.render_mode = render_mode
        self.dt = config.get('dt', .2)
        self.problem = _make_problem(self, config)

        self.time = 0.

        self._objects: Dict[str, Object] = {}
        self._agents: Dict[str, Agent] = {}
        self.metrics: Dict[str, Metric] = {}
        self.possible_agents = ReadOnlyList(self.problem.possible_agents)

        self.render_mode = render_mode
        self._renderer = make_env_renderer(self, render_mode, render_kwargs)

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    @property
    def objects(self) -> Dict[str, Object]:
        return ReadOnlyDict(self._objects)

    @property
    def agents(self):
        return list(self._agents.keys())

    def get_agent(self, name):
        return self._agents[name]

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.problem.action_space(agent)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.problem.observation_space(agent)

    def make_sensors(self, proxy, key='sensors'):
        """
        Create sensors as defined in the configuration.
        """
        from .Env import make_sensors
        return make_sensors(self, proxy, self._config[key])

    def add_object(self, key: str, value: Object):
        if key in self._objects:
            raise KeyError(f"Object with key {key} already exists")

        if key in self.possible_agents:
            raise RuntimeError(f"Object with key {key} is registered as a possible agent")

        self._objects[key] = value

    def pop_object(self, key: str):
        return self._objects.pop(key, None)

    def get_or_create_metric(self, key, producer):
        try:
            return self.metrics[key]
        except KeyError:
            m = producer()
            self.metrics[key] = m
            return m

    def add_agent(self, key: str, value: Agent):
        if key not in self.possible_agents:
            raise KeyError(f"Key {key} is not known as a possible agent")

        if key in self._agents:
            raise KeyError(f"Agent with key {key} already exists")

        assert key not in self._objects

        self._agents[key] = value
        self._objects[key] = value.proxy

    def get_available_agent(self) -> Optional[str]:
        for agent in self.possible_agents:
            if agent not in self._objects:
                return agent
        return None
    
    def render(self):
        return self._renderer.render_manual()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict[AgentID, ObsType],
                                                                                   dict[AgentID, dict]]:

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self.time = 0.
        # Add some global state objects
        self._objects = {'tire_slip': TireSlipObject()}
        self._agents = {}
        self.metrics = {}

        self.problem.configure_env(self.np_random)

        obs, info = self._make_obs()

        self._renderer.render_automatic()

        return obs, info

    def step(self, actions: dict) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        assert set(actions.keys()) == set(self.agents)

        self.time += self.dt

        for k, v in actions.items():
            s_control, l_control = self.problem.action(k).interpret(v)
            control = np.concatenate([np.array([s_control]), l_control], -1)

            self._agents[k].proxy.model.update(control, self.dt)

        for obj_key, obj_val in self.objects.items():
            obj_val.update(self)

        self.problem.update(self.dt)

        # Observe first
        obs, info = self._make_obs()

        self._renderer.render_automatic()

        # Remove terminated agents
        terminated, truncated, reward = {}, {}, {}

        agents_to_remove = []

        for k, v in self._agents.items():
            ag_reward, ag_terminated, ag_truncated, ag_info = v.get_and_clear_pending()
            terminated[k] = ag_terminated
            truncated[k] = ag_truncated
            reward[k] = ag_reward
            info[k] = {**info[k], **ag_info}

            if ag_terminated or ag_truncated:
                agents_to_remove.append(k)

        for ag in agents_to_remove:
            self._agents.pop(ag)
            self._objects.pop(ag)

        # Add global infos under `None` key
        info[None] = {f"Metric.{k}": v.report() for k, v in self.metrics.items()}

        return obs, reward, terminated, truncated, info

    def _make_obs(self) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        agent_obs, agent_infos = {}, {}

        for agent_name, agent in self._agents.items():
            agent_infos[agent_name] = {}
            agent_obs[agent_name] = {
                sensor_name: sensor.observe() for sensor_name, sensor in agent.sensors.items()
            }

        return agent_obs, agent_infos

from typing import Optional, Tuple, Any, Dict
from enum import IntEnum
from copy import deepcopy

import gymnasium as gym

import numpy as np

from .ReadOnlyCollections import ReadOnlyDict
from .Metrics import Metric
from .Object import Object
from .Sensor import Sensor
from .Agent import Agent
from .Rendering.EnvRenderer import make_env_renderer


class DiscreteSteerAction(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


def parse_generic(token: str):
    func_start = token.index('(')
    func_stop = token.index(')')

    assert func_stop == len(token) - 1

    func_name = token[:func_start]
    arg_start = func_start + 1
    arg_stop = func_stop

    if arg_start < arg_stop:
        args = token[arg_start:arg_stop].split(',')
    else:
        args = []

    return func_name, args


def make_action(env, action_dict):
    from .Actions import ContinuousSteeringAction, HumanContinuousSteeringAccelAction, ContinuousSteeringAccelAction
    from .Actions import HumanContinuousSteeringPedalsAction, ContinuousSteeringPedalsAction
    from .Actions import HumanContinuousSteeringAccelToggleAction

    known_types = {
        'continuous_steering': ContinuousSteeringAction,
        'continuous_steering_pedals': ContinuousSteeringPedalsAction,
        'continuous_steering_accel': ContinuousSteeringAccelAction,
        'human': HumanContinuousSteeringAccelAction,
        'human_pedals': HumanContinuousSteeringPedalsAction,
        'human_toggle': HumanContinuousSteeringAccelToggleAction,
    }

    kwargs = {k: v for k, v in action_dict.items() if k != 'type'}

    return known_types[action_dict['type']](env, **kwargs)


def make_problem(env, config: dict):
    from .Problems import FreeDriveProblem, ParallelParkingProblem, RacingProblem, IntersectionProblem

    params = config.get('problem', {'type': 'freedrive'})

    known_types = {
        'freedrive': FreeDriveProblem,
        'intersection': IntersectionProblem,
        'parallel_parking': ParallelParkingProblem,
        'racing': RacingProblem,
    }

    kwargs = {k: v for k, v in params.items() if k != 'type'}
    return known_types[params['type']](env, **kwargs)


def make_sensors(env, proxy, sensor_mapping):
    from .SensorConeMap import SensorConeMap
    from .SensorVehicles import SensorVehicles
    from .SensorTrajectory import SensorTrajectory

    known_types = {
        'conemap': SensorConeMap,
        'trajectory': SensorTrajectory,
        'vehicles': SensorVehicles,
    }

    sensors = {}

    for s_key, s_val in sensor_mapping.items():
        sensor_type = s_val['type']
        kwargs = {k: v for k, v in s_val.items() if k != 'type'}

        sensors[s_key] = known_types[sensor_type](env, proxy, **kwargs)

    return sensors


def parse_vehicle(config: dict, key='vehicle'):
    params = config.get(key, {'type': 'simple', 'wheelbase': 2.4, 'mass': 700.})
    params['physics_divider'] = config.get('physics_divider', 1)

    kwargs = {k: v for k, v in params.items() if k != 'type'}
    return params['type'], kwargs


def make_vehicle(config: dict, key='vehicle'):
    veh_model, vm_kwargs = parse_vehicle(config, key=key)

    if veh_model == 'bicycle':
        from .Physics.BicycleModel import BicycleModel
        return BicycleModel(**vm_kwargs)
    elif veh_model == 'dyn_dugoff':
        from .Physics.SingleTrackDugoffModel import SingleTrackDugoffModel
        return SingleTrackDugoffModel(**vm_kwargs)
    else:
        raise ValueError(f"{veh_model = }")


class CarEnv(gym.Env):
    metadata = {
        'render_modes': ["human", "rgb_array"]
    }

    def __init__(self, config=None, render_mode=None, render_kwargs=None):
        super(CarEnv, self).__init__()

        self._np_random = np.random.default_rng()
        self._config = deepcopy(config) if config is not None else {}
        self._action = self._make_action()
        self.action_space = self._action.action_space

        self.dt = config.get('dt', .2)
        self.problem = make_problem(self, self._config)

        self._reset_required = True

        self._objects: Dict[str, Object] = {}
        self.metrics: Dict[str, Metric] = {}

        self.agent = None

        self.collision_bb = self._config.get('collision_bb', (-1.5, 1.5, -0.8, 0.8))
        self.steering_history_length = 20

        self.render_mode = render_mode
        self._renderer = make_env_renderer(self, render_mode, render_kwargs)

        # Statistics
        self.steps = 0
        self.time = 0

        # Query sensors for obs space, do this last such that everything initialized
        obs_space = {
            "state": self.problem.state_observation_space,
        }

        for s_key, s_val in self._make_sensors(None).items():
            obs_space[s_key] = s_val.observation_space

        self.observation_space = gym.spaces.Dict(obs_space)

    def render(self):
        return self._renderer.render_manual()

    @property
    def vehicle_model(self):
        return self.agent.proxy.model

    @property
    def objects(self) -> Dict[str, Object]:
        return ReadOnlyDict(self._objects)

    def add_object(self, key: str, value: Object):
        if key in self._objects:
            raise KeyError(f"Object with key {key} already exists")

        self._objects[key] = value

    @property
    def agents(self):
        """
        Get the name of currently active agents, for compatibility with multi agent env interface
        """
        return ["ego_proxy"]

    def get_agent(self, name):
        if name != "ego_proxy":
            raise ValueError(f"{name = }")

        return self.agent

    @property
    def sensors(self) -> Dict[str, Sensor]:
        return self.agent.sensors

    def add_to_reward(self, val):
        self.agent.add_to_reward(val)

    def add_info(self, key, val):
        self.agent.add_info(key, val)

    def get_or_create_metric(self, key, producer):
        try:
            return self.metrics[key]
        except KeyError:
            m = producer()
            self.metrics[key] = m
            return m

    def set_reward(self, val):
        self.agent.set_reward(val)

    def set_terminated(self, val):
        self.agent.set_terminated(val)

    def set_truncated(self, val):
        self.agent.set_truncated(val)

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        assert not self._reset_required

        # Interpret action
        s_control, l_control = self._action.interpret(action)

        self.steps += 1
        self.time += self.dt

        control = np.concatenate([np.array([s_control]), l_control], -1)

        self.vehicle_model.update(control, self.dt)

        # Update objects, check cone collision
        for obj_key, obj_val in self.objects.items():
            obj_val.update(self)

        obs = self._make_obs()

        problem_terminated, problem_truncated = self.problem.update(self.dt)

        if problem_terminated:
            self.agent.set_terminated(True)

        if problem_truncated:
            self.agent.set_truncated(True)

        agent_reward, agent_terminated, agent_truncated, agent_info = self.agent.get_and_clear_pending()

        # gymnasium-API calls to always render if render_mode set to human
        self._renderer.render_automatic()

        info = {
            **agent_info,
            **{f"Metric.{k}": v.report() for k, v in self.metrics.items()},
        }

        return obs, agent_reward, agent_terminated, agent_truncated, info

    @property
    def action(self):
        return self._action

    @property
    def ego_pose(self):
        return self.vehicle_model.pose

    @property
    def ego_transform(self):
        return self.agent.proxy.transform

    @property
    def ego_collider(self):
        return self.agent.proxy.collider

    @property
    def last_observations(self):
        # Deprecated but included for old code
        return self._make_obs()

    def _make_obs(self):
        obs = {
            'state': self.problem.observe_state(),
        }

        for s_key, s_val in self.sensors.items():
            obs[s_key] = s_val.observe()

        return obs

    def _make_action(self):
        return make_action(self, self._config.get('action', {'type': 'continuous_steering'}))

    def _make_sensors(self, proxy):
        return make_sensors(self, proxy, self._config.get('sensors', {}))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        from .VehicleProxy import VehicleProxy
        from .TireSlipObject import TireSlipObject

        super(CarEnv, self).reset(seed=seed)

        if self._renderer is not None:
            self._renderer.reset()

        # velocity_controller = LinearVelocityController(130 / 3.6, 9.81, 9.81)
        vehicle_model = make_vehicle(self._config)

        self.agent = Agent(VehicleProxy(self, vehicle_model, self.collision_bb))

        self.vehicle_model.reset()

        self.metrics = {}
        self._objects = {'ego_proxy': self.agent.proxy, 'tire_slip': TireSlipObject()}

        pose = self.problem.configure_env(rng=self._np_random)

        self.agent.sensors = self._make_sensors(self.agent.proxy)

        self.vehicle_model.set_pose(pose)

        self._reset_required = False
        self.steps = 0
        self.time = 0.

        # Make obs before rendering to make last_observation accessible
        reset_obs = self._make_obs()

        # gymnasium-API calls to always render if render_mode set to human
        self._renderer.render_automatic()

        return reset_obs, {f"Metric.{k}": v.report() for k, v in self.metrics.items()}

import sys
from typing import List

import gymnasium as gym
from shapely.geometry import LineString, CAP_STYLE
from shapely.ops import unary_union
import numpy as np

from .MultiAgentProblem import MultiAgentProblem
from .LaneletProblem import VehicleTracker, start_pose
from .IntersectionProblem import intersection_routes, traffic_driving_penalty
from ..Agent import Agent
from ..VehicleProxy import VehicleProxy
from ..SensorComposed import SensorComposed
from ..SensorDynamicOdometry import SensorDynamicOdometry
from ..SensorTrajectory import SensorTrajectory
from ..Env import make_action
from ..Metrics import CounterMetric, EpisodeAveragingMetric


def spawn_groups_from_routes(route_names):
    choices = {}

    for k in route_names:
        n_from, n_to = k.split('->')
        choices[n_from] = choices.get(n_from, []) + [k]

    return choices


class Spawner:
    def __init__(self, route_names, collider):
        self.route_names = route_names
        self.collider = collider
        self.last_spawn = None

    def should_spawn(self, env):
        if self.last_spawn is not None and self.last_spawn + 1.5 >= env.time:
            return False

        for ag in env.agents:
            if env.get_agent(ag).proxy.collider.intersects(self.collider):
                return False
        return True


def handle_vehicle_vehicle_collisions(env, agents, dt, poses=None, soft_collision_pad=None, k_soft_collision=.5):
    if poses is None:
        poses = [agent.proxy.model.pose for agent in agents]

    # Fast agent-agent collision pre-check
    agent_positions = np.asarray(poses)[:, :2] if poses else np.empty((0, 2))
    dists = np.sum(np.square(agent_positions - agent_positions[:, None]), axis=-1)

    for i, j in zip(*np.where(dists < 10. ** 2)):
        if i >= j:
            continue

        ag_i = agents[i]
        ag_j = agents[j]

        if soft_collision_pad is not None:
            if ag_i.proxy.extended_collider(*soft_collision_pad).intersects(
                    ag_j.proxy.extended_collider(*soft_collision_pad)):
                ag_i.add_to_reward(-k_soft_collision * dt)
                ag_j.add_to_reward(-k_soft_collision * dt)
            else:
                # No need to check hard collision without soft collision
                continue

        if ag_i.proxy.collider.intersects(ag_j.proxy.collider):
            ag_i.set_terminated(True)
            ag_i.set_reward(0. if ag_i.proxy.model.is_nearly_stationary else -1.)
            ag_i.add_info('Done.Reason', 'Collided')
            ag_j.set_terminated(True)
            ag_j.set_reward(0. if ag_j.proxy.model.is_nearly_stationary else -1.)
            ag_j.add_info('Done.Reason', 'Collided')
            env.get_or_create_metric('CollidedVehicles', CounterMetric).increment(2)


class MultiAgentLaneletProblem(MultiAgentProblem):
    def __init__(self, env, route_dict, max_vehicles=30, k_forwards=.05, k_tracking=.02, lookahead_points=30,
                 soft_collision_pad=None, k_soft_collision=.5, spawn_max_delta=.2, spawn_velocity_range=(5.55, 8.33)):
        super(MultiAgentLaneletProblem, self).__init__(env)
        self.route_dict = route_dict
        self.max_vehicles = max_vehicles
        self.lookahead_points = lookahead_points
        self._spawners = []
        self._possible_agents = [f"veh_{i}" for i in range(30)]

        for rn in spawn_groups_from_routes(self.route_dict.keys()).values():
            ls = LineString(self.route_dict[rn[0]])
            coords = np.stack([start_pose(ls, 2.5), start_pose(ls, 7.5)])
            collider = LineString(coords[:, :2]).buffer(1., cap_style=CAP_STYLE.square)
            self._spawners.append(Spawner(rn, collider))

        self.k_forwards = k_forwards
        self.k_tracking = k_tracking
        self.k_soft_collision = k_soft_collision
        self.soft_collision_pad = soft_collision_pad
        self.spawn_max_delta = spawn_max_delta
        self.spawn_velocity_range = spawn_velocity_range

        polies = []

        for k, v in self.route_dict.items():
            ls = LineString(v)
            polies.append(ls.buffer(2.5, cap_style=CAP_STYLE.flat))

        self.road_poly = unary_union(polies)
        self._view_box = self.road_poly.bounds

        # Spaces (same for all agents)
        self._action_space = gym.spaces.Box(-1., 1., (3,))
        self._observation_space = gym.spaces.Dict({s_key: s_val.observation_space for s_key, s_val in self._make_sensors(None).items()})
        self._action = make_action(env, env._config['action'])

    @property
    def possible_agents(self) -> List[str]:
        return self._possible_agents

    def transform_ctx(self, ctx, width, height):
        from ..Rendering.BirdView import transform_for_bbox
        transform_for_bbox(ctx, width, height, self._view_box)

    def action(self, agent):
        return self._action

    def action_space(self, agent):
        return self._action_space

    def observation_space(self, agent):
        return self._observation_space

    def _make_sensors(self, proxy):
        sensors = self.env.make_sensors(proxy)
        sensors['state'] = SensorComposed(self.env, proxy,
                                          SensorDynamicOdometry(self.env, proxy),
                                          SensorTrajectory(self.env, proxy, self.lookahead_points))
        return sensors

    def draw(self, draw):
        from ..Rendering.DrawLevels import LEVEL_GROUND, LEVEL_REF_TRACK
        draw.defer(LEVEL_GROUND, self._deferred_draw_ground)
        draw.defer(LEVEL_REF_TRACK, self._deferred_draw_ui)

    def _deferred_draw_ground(self, ctx):
        from ..Rendering.TrackRenderer import trace_path
        from ..Rendering.Rendering import stroke_fill

        trace_path(ctx, self.road_poly.exterior.coords)
        for interior in self.road_poly.interiors:
            trace_path(ctx, interior.coords)

        stroke_fill(ctx, (0., 0., 0.), (.6, .6, .6))

    def _deferred_draw_ui(self, ctx):
        from ..Rendering.Rendering import stroke_fill

        # Only uncomment spawn areas for debugging (or make toggleable)
        # for s in self._spawners:
        #     ctx.new_sub_path()
        #     for x, y in s.collider.exterior.coords:
        #         ctx.line_to(x, y)
        #     ctx.close_path()
        # stroke_fill(ctx, (0., 0., 0.), None)

        for ag in self.env.agents:
            agent = self.env.get_agent(ag)

            ctx.new_path()
            for x, y in agent.problem_data['tracker'].trajectory.coords:
                ctx.line_to(x, y)

            stroke_fill(ctx, agent.proxy.color, None)

    def _spawn(self, route_id):
        from ..Rendering.Colors import random_vehicle_color
        from ..Env import make_vehicle

        agent_name = self.env.get_available_agent()

        if agent_name is None:
            print("No available agent", file=sys.stderr)
            return

        tracking_ls = LineString(self.route_dict[route_id])
        pose = start_pose(tracking_ls, start_offset=5.)
        pose[:2] += self.env.np_random.uniform(-self.spawn_max_delta, self.spawn_max_delta, size=(2,))
        tracker = VehicleTracker(tracking_ls, pose, self.k_forwards, self.k_tracking)

        # TODO: Better way of spawning?
        vehicle_model = make_vehicle(self.env._config)
        agent = Agent(VehicleProxy(self.env, vehicle_model, self.env._config.get('collision_bb', (-1.5, 1.5, -0.8, 0.8)),
                                   color=random_vehicle_color(self.env.np_random)))
        vehicle_model.reset()
        vehicle_model.set_pose(pose)
        vehicle_model.set_velocity(self.env.np_random.uniform(*self.spawn_velocity_range))

        agent.sensors = self._make_sensors(agent.proxy)

        agent.problem_data = {
            'route_id': route_id,
            'tracker': tracker,
            'trajectory': tracking_ls,
        }

        self.env.add_agent(agent_name, agent)

    def trajectory_for_proxy(self, proxy):
        for ag in self.env.agents:
            agent = self.env.get_agent(ag)

            if agent.proxy == proxy:
                return agent.problem_data['trajectory']
        raise KeyError

    def configure_env(self, rng=None):
        # Do not draw tire slip
        self.env.pop_object('tire_slip')

        rng.shuffle(self._spawners)
        for s in self._spawners:
            if len(self.env.agents) < self.max_vehicles:
                s.last_spawn = self.env.time
                self._spawn(rng.choice(s.route_names))

        # Create metrics
        self.env.get_or_create_metric('TripsCompleted', CounterMetric)
        self.env.get_or_create_metric('ComeOffVehicles', CounterMetric)
        self.env.get_or_create_metric('CollidedVehicles', CounterMetric)
        self.env.get_or_create_metric('MeanVelocity', EpisodeAveragingMetric)

    def update(self, dt: float):
        agents = [self.env.get_agent(ag) for ag in self.env.agents]

        poses = []

        v_metric = self.env.metrics['MeanVelocity']

        for agent in agents:
            # Store pose for later
            pose = agent.proxy.model.pose
            poses.append(pose)
            tracker = agent.problem_data['tracker']
            reward, reached_goal = tracker.update(pose, dt)
            v_metric.update(tracker.moved_distance_ / dt)
            agent.add_to_reward(reward)
            agent.add_to_reward(-traffic_driving_penalty(agent.proxy.model, slip_start_deg=4.) * dt)

            if reached_goal:
                agent.set_reward(1.)
                agent.set_terminated(True)
                agent.add_info('Done.Reason', 'CompletedRoute')
                self.env.get_or_create_metric('TripsCompleted', CounterMetric).increment()

        # Check in bounds
        for i, agent in enumerate(agents):
            if not self.road_poly.contains(agent.proxy.collider):
                agent.set_terminated(True)
                agent.set_reward(-1.)
                agent.add_info('Done.Reason', 'LeftRoad')
                self.env.get_or_create_metric('ComeOffVehicles', CounterMetric).increment()

        handle_vehicle_vehicle_collisions(self.env, agents, dt, poses, soft_collision_pad=self.soft_collision_pad,
                                          k_soft_collision=self.k_soft_collision)

        # If we can spawn more vehicles, shuffle the spawners (so we don't select the same first) and try spawning
        if len(self.env.agents) < self.max_vehicles:
            self.env.np_random.shuffle(self._spawners)
            for s in self._spawners:
                if s.should_spawn(self.env):
                    self._spawn(self.env.np_random.choice(s.route_names))
                    s.last_spawn = self.env.time + self.env.np_random.normal(loc=0., scale=.2)
                    if len(self.env.agents) >= self.max_vehicles:
                        break


class MultiAgentIntersectionProblem(MultiAgentLaneletProblem):
    def __init__(self, env, **kwargs):
        super(MultiAgentIntersectionProblem, self).__init__(env, intersection_routes(), **kwargs)

from typing import Tuple, Union

import cairo
import numpy as np

from .Problem import Problem
from .RacingProblem import encode_odometry
from ..Metrics import CounterMetric
import gymnasium as gym
from shapely.geometry import LineString, LinearRing, Point, CAP_STYLE
from shapely.ops import unary_union


def spline_lookahead(ls: Union[LinearRing, LineString], pose, steps):
    x, y, theta = pose
    a = ls.project(Point([x, y]))

    result = []

    wrap = isinstance(ls, LinearRing)

    for s in steps:
        t = a + s

        if wrap:
            t = t % ls.length

        result.append(ls.interpolate(t).coords)

    return np.concatenate(result)


def start_pose(ls: LineString, start_offset=5.):
    (x1, y1), = ls.interpolate(start_offset).coords
    (x2, y2), = ls.interpolate(start_offset + 1.).coords

    return np.array([x1, y1, np.arctan2(y2 - y1, x2 - x1)])


class VehicleTracker:
    def __init__(self, trajectory, initial_pose, k_forwards, k_tracking, v_max=None):
        self.trajectory = trajectory
        self._last_position = initial_pose[:2]
        self._last_project = self.trajectory.project(Point(self._last_position))
        self.k_forwards = k_forwards
        self.k_tracking = k_tracking
        self.v_max = v_max
        self.tracking_distance_ = 0.
        self.moved_distance_along_track_ = 0.
        self.moved_distance_ = 0.

    def update(self, pose, dt):
        new_position = pose[:2]
        ego_point = Point(new_position)
        new_project = self.trajectory.project(ego_point)
        self.moved_distance_ = np.linalg.norm(new_position - self._last_position)
        self.moved_distance_along_track_ = new_project - self._last_project

        if isinstance(self.trajectory, LinearRing):
            if self.moved_distance_along_track_ > self.trajectory.length / 2:
                # Moved backwards through loop-close
                self.moved_distance_along_track_ = new_project - (self._last_project + self.trajectory.length)
            elif self.moved_distance_along_track_ < -self.trajectory.length / 2:
                # Moved forwards through loop-close
                self.moved_distance_along_track_ = (new_project + self.trajectory.length) - self._last_project

        self.tracking_distance_ = ego_point.distance(self.trajectory)

        if self.v_max is None:
            mdat = self.moved_distance_along_track_
        else:
            # Backwards deliberately unbounded
            mdat = min(self.moved_distance_along_track_, dt * self.v_max)

        reward = mdat * self.k_forwards - self.tracking_distance_ * \
                 self.moved_distance_ * self.k_tracking
        reached_goal = new_project > self.trajectory.length - 5.

        self._last_project = new_project
        self._last_position = new_position

        return reward, reached_goal


class LaneletProblem(Problem):
    def __init__(self, env, route_dict, lookahead_points=10, max_time=20., k_forwards=.05, k_tracking=.02):
        super(LaneletProblem, self).__init__(env)
        self.route_dict = route_dict
        self.lookahead_points = lookahead_points
        self.road_poly = None
        self.tracking_route = None
        self.tracking_ls = None
        self._tracker = None
        self.time = None
        self.max_time = max_time
        self.k_forwards = k_forwards
        self.k_tracking = k_tracking

        polies = []

        for k, v in self.route_dict.items():
            ls = LineString(v)
            polies.append(ls.buffer(2.5, cap_style=CAP_STYLE.flat))

        self.road_poly = unary_union(polies)

    @property
    def state_observation_space(self):
        return gym.spaces.Box(-1, 1, (6 + self.lookahead_points * 2,))

    def configure_env(self, rng=None) -> Tuple[float, float, float]:
        self.tracking_route = rng.choice(list(self.route_dict.keys()))
        self.tracking_ls = LineString(self.route_dict[self.tracking_route])
        self.time = 0.
        pose = start_pose(self.tracking_ls, start_offset=5.)
        self._tracker = VehicleTracker(self.tracking_ls, pose, self.k_forwards, self.k_tracking)
        return pose

    def observe_state(self):
        env = self.env

        spline_data = spline_lookahead(self.tracking_ls, env.ego_pose, list(range(self.lookahead_points)))

        spline_data_local = env.ego_transform @ np.concatenate([spline_data, np.ones_like(spline_data[:, :1])], axis=-1).T
        spline_data_local = spline_data_local.T[:, :2]

        # Normalize
        spline_data_local = spline_data_local / self.lookahead_points

        return np.concatenate([encode_odometry(env.vehicle_model), spline_data_local.reshape(-1)])

    def update(self, dt: float) -> Tuple[bool, bool]:
        env = self.env

        terminate = False

        self.time += dt
        truncate = self.time > self.max_time

        tracking_reward, reached_goal = self._tracker.update(env.ego_pose, dt)
        env.add_to_reward(tracking_reward)

        env.get_or_create_metric('TrackingError', CounterMetric).increment(self._tracker.tracking_distance_ *
                                                                           self._tracker.moved_distance_along_track_)

        if not self.road_poly.contains(env.ego_collider):
            terminate = True
            env.set_reward(-1.)
            env.add_info('Done.Reason', 'LeftRoad')
        elif reached_goal:
            terminate = True
            env.set_reward(1.)
            env.add_info('Done.Reason', 'CompletedRoute')

        return terminate, truncate

    def draw(self, draw):
        from ..Rendering.DrawLevels import LEVEL_GROUND, LEVEL_SENSOR_VISUALIZATION
        draw.defer(LEVEL_GROUND, self._deferred_draw_ground)
        draw.defer(LEVEL_SENSOR_VISUALIZATION, self._deferred_draw_ui)

    def _deferred_draw_ui(self, ctx: cairo.Context):
        from ..Rendering.Rendering import stroke_fill

        ctx.new_path()
        for x, y in self.tracking_ls.coords:
            ctx.line_to(x, y)

        stroke_fill(ctx, (1., 1., 1.), None)

    def _deferred_draw_ground(self, ctx):
        from ..Rendering.TrackRenderer import trace_path
        from ..Rendering.Rendering import stroke_fill

        trace_path(ctx, self.road_poly.exterior.coords)
        for interior in self.road_poly.interiors:
            trace_path(ctx, interior.coords)

        stroke_fill(ctx, (0., 0., 0.), (.6, .6, .6))

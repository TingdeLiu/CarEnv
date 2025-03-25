import gymnasium as gym
import numpy as np

from .Problems.LaneletProblem import spline_lookahead
from .Sensor import Sensor


class SensorTrajectory(Sensor):
    """
    A trajectory sensor. Can only be used with trajectory based problems.
    """

    def __init__(self, env, proxy, lookahead_points=50, step=1.):
        super().__init__(env, proxy)

        self.lookahead_points = lookahead_points
        self.step = step
        self.trajectory = None

    @property
    def normalizer(self):
        return np.sqrt(2) * self.step * self.lookahead_points

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(-1, 1, (self.lookahead_points * 2,))

    def _observe_impl(self):
        # Lazy init, to ensure we are in the scene
        if self.trajectory is None:
            # Query the trajectory for our vehicle proxy from the problem (The problem must support this)
            self.trajectory = self.env.problem.trajectory_for_proxy(self.proxy)

        steps = np.arange(self.lookahead_points) * self.step
        spline_data = spline_lookahead(self.trajectory, self.proxy.model.pose, steps)
        spline_data_local = self.proxy.transform @ np.concatenate([spline_data, np.ones_like(spline_data[:, :1])],
                                                               axis=-1).T
        spline_data_local = spline_data_local.T[:, :2]

        # Normalize
        spline_data_local = spline_data_local / self.normalizer
        # And clip, because of the possible offset of the ego vehicle from the trajectory points could otherwise
        # be slightly outside of the [-1, 1] range
        spline_data_local = np.clip(spline_data_local, -1, 1)

        return spline_data_local.reshape(-1)

    def draw(self, ctx):
        from .Util import inverse_transform_from_pose
        from .Rendering.Rendering import stroke_fill

        trajectory_pts = self.last_observation.reshape(-1, 2) * self.normalizer
        trajectory_pts = np.concatenate([trajectory_pts, np.ones_like(trajectory_pts[:, :1])], axis=-1)

        global_pts = (inverse_transform_from_pose(self.proxy.model.pose) @ trajectory_pts.T).T

        for x, y, _ in global_pts:
            ctx.new_sub_path()
            ctx.arc(x, y, .2, 0., 2. * np.pi)
        stroke_fill(ctx, (0., 0., 0.), self.proxy.color if self.proxy.color is not None else (0., 0., 0.))

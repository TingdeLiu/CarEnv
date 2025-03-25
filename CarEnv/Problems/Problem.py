from typing import Tuple


class Problem:
    def __init__(self, env):
        self.env = env

    @property
    def state_observation_space(self):
        raise NotImplementedError

    def observe_state(self):
        raise NotImplementedError

    def configure_env(self, rng=None) -> Tuple[float, float, float]:
        raise NotImplementedError

    def transform_ctx(self, ctx, width, height):
        from ..Rendering.BirdView import transform_for_pose
        scale = height / 57.5
        transform_for_pose(ctx, width, height, self.env.ego_pose, scale=scale)

    def draw(self, draw):
        pass

    def update(self, dt: float) -> Tuple[bool, bool]:
        """
        Update the environment for the problem
        Args:
            dt: Time delta in seconds

        Returns:
            terminate, truncate
        """
        raise NotImplementedError

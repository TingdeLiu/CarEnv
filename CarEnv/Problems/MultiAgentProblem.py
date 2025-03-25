from typing import List


class MultiAgentProblem:
    """
    Base class for multi-agent problems
    """

    def __init__(self, env):
        self.env = env

    @property
    def possible_agents(self) -> List[str]:
        raise NotImplementedError

    def action(self, agent):
        raise NotImplementedError

    def action_space(self, agent):
        raise NotImplementedError

    def observation_space(self, agent):
        raise NotImplementedError

    def configure_env(self, rng=None):
        raise NotImplementedError

    def draw(self, draw):
        pass

    def transform_ctx(self, ctx, width, height):
        from ..Rendering.BirdView import transform_fixed_point
        transform_fixed_point(ctx, width, height, (0., 0.))

    def update(self, dt: float):
        raise NotImplementedError

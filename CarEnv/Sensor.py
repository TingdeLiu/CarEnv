import gymnasium as gym


class Sensor:
    def __init__(self, env, proxy):
        """
        Create a sensor instance in the given environment attached to the given proxy.
        :param env: Environment this sensor lives in
        :param proxy: (Vehicle) proxy this sensor is attached to. The sensor implementation *must* support being
        initialized with `None` as proxy and return the same observation space as with any valid proxy.
        """
        self.env = env
        self.proxy = proxy
        self.last_observation = None

    @property
    def observation_space(self) -> gym.Space:
        raise NotImplementedError

    def observe(self):
        self.last_observation = self._observe_impl()
        return self.last_observation

    def _observe_impl(self):
        raise NotImplementedError

    def draw(self, ctx):
        pass

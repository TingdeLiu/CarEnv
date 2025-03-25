import gymnasium as gym
import numpy as np

from .Sensor import Sensor


class SensorComposed(Sensor):
    def __init__(self, env, proxy, *args):
        super(SensorComposed, self).__init__(env, proxy)
        self.sensors = args

    @property
    def observation_space(self) -> gym.Space:
        shape, low, high = 0, [], []

        for s in self.sensors:
            obs_space = s.observation_space

            if not isinstance(obs_space, gym.spaces.Box):
                raise TypeError(type(obs_space))

            n, = obs_space.shape
            shape += n
            low.append(np.broadcast_to(obs_space.low, (n,)))
            high.append(np.broadcast_to(obs_space.high, (n,)))

        return gym.spaces.Box(np.concatenate(low), np.concatenate(high), (shape,))

    def _observe_impl(self):
        return np.concatenate([s.observe() for s in self.sensors])

    def draw(self, ctx):
        for s in self.sensors:
            s.draw(ctx)

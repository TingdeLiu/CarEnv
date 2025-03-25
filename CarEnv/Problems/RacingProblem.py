import gymnasium as gym
from .FreeDriveProblem import FreeDriveProblem
from ..SensorDynamicOdometry import encode_odometry


class RacingProblem(FreeDriveProblem):
    @property
    def state_observation_space(self):
        return gym.spaces.Box(-1., 1., (6,))

    def observe_state(self):
        return encode_odometry(self.env.vehicle_model)

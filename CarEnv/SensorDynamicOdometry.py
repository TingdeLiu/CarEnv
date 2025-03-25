import gymnasium as gym
import numpy as np

from .Sensor import Sensor


def encode_odometry(vehicle_model):
    omega, beta = vehicle_model.omega_, vehicle_model.steering_angle

    v_x, v_y = vehicle_model.v_loc_ if vehicle_model.v_loc_ is not None else (0, 0)

    if hasattr(vehicle_model, 'omega_front_'):
        omega_front, omega_rear = vehicle_model.omega_front_, vehicle_model.omega_rear_
        # Slight safety-factor such that slip at maximum speed is not clipped
        omega_front /= vehicle_model.max_angular_velocity * 1.1
        omega_rear /= vehicle_model.max_angular_velocity * 1.1
    else:
        omega_front, omega_rear = 0., 0.

    result = np.array([
        v_x / vehicle_model.top_speed,
        v_y / vehicle_model.top_speed,
        omega / 6.28,  # Maximum encodable angular velocity of vehicle is ~360°/sec
        beta / vehicle_model.steering_model.beta_max,
        omega_front,
        omega_rear,
    ], dtype=np.float32)

    clipped_result = np.clip(result, -1., 1.)

    if np.any(clipped_result != result):
        from warnings import warn
        warn(f"Some values of the observation were clipped to fit the observation space: {result}")

    return clipped_result


class SensorDynamicOdometry(Sensor):
    def __init__(self, env, proxy):
        super(SensorDynamicOdometry, self).__init__(env, proxy)

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(-1., 1., (6,))

    def _observe_impl(self):
        return encode_odometry(self.proxy.model)

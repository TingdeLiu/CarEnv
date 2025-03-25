from .SteeringController import RateLimitedSteeringModel
from .AbstractCarModel import AbstractCarModel
from typing import Tuple
import numpy as np


class BicycleModel(AbstractCarModel):
    """
    A kinematic single track model with limited acceleration, deceleration and steering rate
    """
    def __init__(self, wheelbase: float, beta_rate=1., beta_max=0.52, engine_power: float = 20000., Acd0: float = 3.,
                 mass: float = 800., brake_force=12000., physics_divider: int = 10):
        # Constants
        self.steering_model = RateLimitedSteeringModel(beta_max, beta_rate)
        self.wheelbase = wheelbase
        self.engine_power = engine_power
        self.Acd0 = Acd0
        self.mass = mass
        self.brake_force = brake_force
        self.physics_divider = physics_divider

        # State
        self.pose_ = None
        self.is_braking_ = None
        self.v_long_ = 0.

    @property
    def steering_angle(self) -> float:
        return self.steering_model.beta

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.v_long_, 0.])

    @property
    def is_braking(self) -> bool:
        return self.is_braking_

    @property
    def pose(self):
        return self.pose_

    @property
    def n_controls(self):
        return 2

    @property
    def turning_circle(self):
        # Turning circle of the center of front axle (which is larger than the rear axle).
        return 2 * self.wheelbase / np.sin(self.steering_model.beta_max)

    def reset(self):
        self.steering_model.reset()
        self.is_braking_ = False
        self.pose_ = np.zeros(3)

    def set_pose(self, pose):
        assert np.shape(pose) == (3,)

        self.pose_ = pose

    def update_velocity(self, control: np.ndarray, dt: float) -> Tuple[float, float]:
        # Small control deadzone
        control = np.where(np.abs(control) < .05, np.zeros_like(control), control)

        # F_W = rho / 2 * c_w * A * v^2
        # rho = 1.25
        force_parasite = 1.25 / 2 * self.Acd0 * np.square(self.v_long_) * -np.sign(self.v_long_)

        # F_R = c_R * F_N = c_R * m * g
        # c_R = 0.013 (car tire on asphalt)
        force_rolling = 4 * 0.013 * self.mass * 9.81 * -np.sign(self.v_long_)

        # P = F * v <=> F = P / v
        # Hack static power
        using_brake = np.sign(control) == -np.sign(self.v_long_)
        force_brake = control * self.brake_force
        force_engine = control * self.engine_power / np.maximum(np.abs(self.v_long_), 2.)
        force_control = np.where(using_brake, force_brake, force_engine)

        # F = m * a <=> a = F / m
        accel = (force_control + force_rolling + force_parasite) / self.mass

        new_v_long = self.v_long_ + accel * dt

        # Fix to zero on zero crossing (otherwise would accelerate using braking force and vice-versa)
        if np.sign(new_v_long) == -np.sign(self.v_long_):
            new_v_long = 0.

        return new_v_long, (new_v_long + self.v_long_) / 2

    def _update_step(self, control: np.ndarray, dt: float):
        if control.shape != (2,):
            raise ValueError(f"Bad control shape, expected {(2,)}, actual {control.shape}")

        s_control, v_control = control

        x, y, theta = self.pose_

        beta_old = self.steering_model.beta

        self.steering_model.integrate(s_control, dt)
        self.v_long_, v_avg = self.update_velocity(v_control, dt)

        self.is_braking_ = np.sign(v_control) == -np.sign(self.v_long_)

        tan_avg = np.tan(beta_old)
        beta = np.arctan(tan_avg / 2)
        delta_x = dt * v_avg * np.cos(theta + beta)
        delta_y = dt * v_avg * np.sin(theta + beta)
        delta_theta = dt * v_avg * np.cos(beta) * tan_avg / self.wheelbase

        self.pose_ = np.array([x + delta_x, y + delta_y, theta + delta_theta])

    def update(self, control: np.ndarray, dt: float):
        for _ in range(self.physics_divider):
            self._update_step(control, dt / self.physics_divider)

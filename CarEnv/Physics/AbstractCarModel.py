import numpy as np


class AbstractCarModel:
    @property
    def pose(self) -> np.ndarray:
        """
        Returns the pose (x, y, theta) in the global frame
        """
        raise NotImplementedError

    @property
    def steering_angle(self) -> float:
        """
        Returns the steering angle of the front axle in radians
        """
        raise NotImplementedError

    @property
    def velocity(self) -> np.ndarray:
        """
        Returns the current velocity vector (v_long, v_lat) in m/s in the reference frame
        """
        raise NotImplementedError

    @property
    def is_nearly_stationary(self) -> bool:
        """
        Returns whether the vehicle is nearly stationary
        """
        return np.linalg.norm(self.velocity) < 1e-2

    @property
    def is_braking(self) -> bool:
        """
        Returns whether the vehicle is braking (for visualization purposes)
        """
        raise NotImplementedError

import gymnasium as gym
import numpy as np

from .Agent import Agent
from .Sensor import Sensor
from .Util import transform_from_pose
from .BatchedObjects import BatchedObjects


class SensorVehicles(Sensor):
    def __init__(self, env, proxy, max_objects=40, bbox=(-30, 30, -30, 30)):
        super(SensorVehicles, self).__init__(env, proxy)
        self.max_objects = max_objects
        self.bbox = bbox

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(-1., 1., (self.max_objects, 7))

    @property
    def view_normalizer(self):
        return max((abs(x) for x in self.bbox))

    @property
    def velocity_normalizer(self):
        # This is a rough estimate, of course other vehicles could be faster
        return 2. * self.proxy.model.top_speed

    def _observe_impl(self):
        poses = []
        velocities = []

        ego_pose = self.proxy.model.pose
        transform = transform_from_pose(ego_pose)

        for ag in self.env.agents:
            agent: Agent = self.env.get_agent(ag)

            if agent.proxy == self.proxy:
                continue

            poses.append(agent.proxy.model.pose)
            velocities.append(agent.proxy.model.velocity)

        poses = np.array(poses) if poses else np.empty((0, 3))
        velocities = np.asarray(velocities) if velocities else np.empty((0, 2))

        pos_hom = np.concatenate([poses[:, :2], np.ones_like(poses[:, :1])], axis=-1)
        new_pos = np.squeeze(transform @ pos_hom[..., None], -1)[:, :2]

        # Filter based on sensor bounding box
        observable = BatchedObjects.filter_mask_aabb(new_pos, *self.bbox)
        new_pos = new_pos[observable]
        velocities = velocities[observable]
        thetas = poses[observable, 2]

        velocities_hom = np.concatenate([velocities, np.zeros_like(velocities[:, :1])], axis=-1)
        new_vel = np.squeeze(transform @ velocities_hom[..., None], -1)[:, :2]
        directions = np.stack([np.cos(thetas - ego_pose[2]), np.sin(thetas - ego_pose[2])], axis=-1)

        data = np.concatenate([new_pos / self.view_normalizer, new_vel / self.velocity_normalizer, directions], axis=-1)
        data = np.clip(data, -1., 1.)

        count = new_pos.shape[0]
        enc_count = min(count, self.max_objects)

        if enc_count < count:
            import warnings
            warnings.warn(f"Discarding {count - enc_count} objects because {self.max_objects = } is too low.")

        result = np.zeros((self.max_objects, 7), dtype=np.float32)
        result[:enc_count, 0] = 1
        result[:enc_count, 1:] = data[:enc_count]

        return result

import cairo
import numpy as np

from .BatchedObjects import BatchedObjects
from .Collision import intersections_aabb_circles, intersection_distances_aabb_circles
from .Metrics import CounterMetric


class BatchedCones(BatchedObjects):
    def __init__(self, data, hit_count=0, soft_collision_distance=None, soft_collision_max_penalty=.2):
        assert len(data.shape) == 2
        assert data.shape[-1] == 4

        # Features are x, y, is_blue, is_yellow
        self._data = data
        self._cached_renderer = None
        self.hit_count = hit_count
        self.soft_collision_distance = soft_collision_distance
        self.soft_collision_max_penalty = soft_collision_max_penalty

    def __with_new_data(self, data):
        return BatchedCones(data, hit_count=self.hit_count, soft_collision_distance=self.soft_collision_distance,
                            soft_collision_max_penalty=self.soft_collision_max_penalty)

    def draw(self, draw):
        from .Rendering.DrawLevels import LEVEL_CONE
        draw.defer(LEVEL_CONE, self.deferred_draw)

    def deferred_draw(self, ctx: cairo.Context):
        if self._cached_renderer is None:
            from .Rendering.ConeRenderer import ConeRenderer
            self._cached_renderer = ConeRenderer()

        types = np.argmax(self.data[:, 2:], axis=-1) + 1
        self._cached_renderer.render(ctx, self.data[:, :2], types, self.radius)

    @property
    def radius(self):
        return .3

    @property
    def centers(self):
        return self._data[:, :2]

    @staticmethod
    def get_cone_color(style):
        if style == 1:
            return 56 / 255, 103 / 255, 214 / 255
        elif style == 2:
            return 254 / 255, 211 / 255, 48 / 255
        else:
            raise ValueError(style)

    @staticmethod
    def categorical_from_indices(types):
        n, = types.shape
        types_idx = types - 1  # Starts at 1
        types_cat = np.zeros((n, 2))
        types_cat[np.arange(n), types_idx] = 1.
        return types_cat

    @staticmethod
    def from_track_dict(track_dict, **kwargs) -> 'BatchedCones':
        pos = track_dict['cone_pos']
        types = track_dict['cone_type']
        return BatchedCones(np.concatenate([pos, BatchedCones.categorical_from_indices(types)], axis=-1), **kwargs)

    @property
    def data(self):
        return self._data

    def transformed(self, transform) -> 'BatchedCones':
        pos_hom = np.concatenate([self._data[:, :2], np.ones_like(self._data[:, :1])], axis=-1)
        new_pos = np.squeeze(transform @ pos_hom[..., None], -1)

        new_data = np.concatenate([new_pos[:, :2], self._data[:, 2:]], axis=-1)
        return self.__with_new_data(new_data)

    def filtered_aabb(self, *args) -> 'BatchedCones':
        mask = BatchedObjects.filter_mask_aabb(self._data[:, :2], *args)
        return self.__with_new_data(self._data[mask])

    def filtered_by_type(self, cone_type):
        """
        Return current cones filtered by type
        :param cone_type: If 1, only return blue cones. If 2, only return yellow cones.
        """
        mask = np.argmax(self.data[:, 2:], axis=-1) == cone_type - 1
        return self.__with_new_data(self._data[mask])

    def update(self, env):
        for agent_name in env.agents:
            agent = env.get_agent(agent_name)

            us_in_local = self.transformed(agent.proxy.transform)

            if self.soft_collision_distance is None:
                intersected = intersections_aabb_circles(agent.proxy.collision_bb, self.radius, us_in_local.data[:, :2])
            else:
                distances = intersection_distances_aabb_circles(agent.proxy.collision_bb,
                                                                self.radius + self.soft_collision_distance,
                                                                us_in_local.data[:, :2])
                distances[~np.isfinite(distances)] = self.radius + self.soft_collision_distance

                intersected = distances < self.radius
                soft_intersected = (distances < self.radius + self.soft_collision_distance) & (~intersected)
                soft_intersection_penalties = (1. - (distances - self.radius) / self.soft_collision_distance) \
                    * soft_intersected * self.soft_collision_max_penalty
                soft_collision_penalty = np.sum(soft_intersection_penalties) * -1
                agent.add_to_reward(soft_collision_penalty)

            hits = np.sum(intersected)
            self.hit_count += hits
            agent.add_to_reward(hits * -.2)
            env.get_or_create_metric('ConesHit', CounterMetric).increment(hits)

            self._data = self._data[~intersected]

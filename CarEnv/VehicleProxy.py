import cairo
import numpy as np

from .DeferredDraw import DeferredDraw
from .Object import Object
from .Util import transform_from_pose, inverse_transform_from_pose
from .Rendering.TireSlipRenderer import TireSlipRenderer


class VehicleProxy(Object):
    def __init__(self, env, model, collision_bb, color=None):
        self.model = model
        self.collision_bb = collision_bb
        self.color = color
        self._tire_slip_renderer = TireSlipRenderer(env)

    @property
    def collider(self):
        from shapely.geometry import Polygon

        xl, xh, yl, yh = self.collision_bb
        pts = (inverse_transform_from_pose(self.model.pose) @ np.array([
            (xl, yl, 1),
            (xl, yh, 1),
            (xh, yh, 1),
            (xh, yl, 1)
        ]).T).T[:, :2]
        return Polygon(pts)

    def extended_collider(self, pad_long, pad_lat):
        from shapely.geometry import Polygon

        xl, xh, yl, yh = self.collision_bb
        pts = (inverse_transform_from_pose(self.model.pose) @ np.array([
            (xl - pad_long, yl - pad_lat, 1),
            (xl - pad_long, yh + pad_lat, 1),
            (xh + pad_long, yh + pad_lat, 1),
            (xh + pad_long, yl - pad_lat, 1)
        ]).T).T[:, :2]
        return Polygon(pts)

    @property
    def transform(self):
        return transform_from_pose(self.model.pose)

    def draw(self, draw: DeferredDraw):
        from .Rendering.DrawLevels import LEVEL_VEHICLE, LEVEL_TIRE_MARKS
        draw.defer(LEVEL_TIRE_MARKS, self._deferred_draw_tires)
        draw.defer(LEVEL_VEHICLE, self._deferred_draw_vehicle)

    def _deferred_draw_tires(self, ctx):
        self._tire_slip_renderer.update(self.model, self.collision_bb[-1] * 2)

    def _deferred_draw_vehicle(self, ctx: cairo.Context):
        from .Rendering.Rendering import draw_old_car

        draw_old_car(ctx, *self.model.pose, self.model.wheelbase, self.model.steering_angle, *self.collision_bb,
                     braking=self.model.is_braking, color=self.color)

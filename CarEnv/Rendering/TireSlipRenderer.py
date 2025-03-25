import numpy as np

from ..Util import inverse_transform_from_pose


class TireSlipRenderer:
    def __init__(self, env):
        self.last_transform = None
        self.env = env

    def _add_slip_line(self, t1, t2, x, y):
        try:
            tire_slip_object = self.env.objects['tire_slip']
        except KeyError:
            return

        vec = np.array([x, y, 1])

        p1 = (t1 @ vec)[:2]
        p2 = (t2 @ vec)[:2]
        tire_slip_object.slip_lines.append((p1, p2))
        tire_slip_object.slip_lines = tire_slip_object.slip_lines[-tire_slip_object.max_lines:]  # Limit to not overwhelm

    def reset(self):
        self.last_transform = None

    def update(self, vehicle_model, track_width):
        if not hasattr(vehicle_model, "front_slip_"):
            # Does not appear to be correct vehicle model
            return

        transform = inverse_transform_from_pose(vehicle_model.pose)

        if self.last_transform is None:
            self.last_transform = transform
            return

        # Return if no physics set yet
        if vehicle_model.front_slip_ is None:
            return

        front_slip = vehicle_model.front_slip_
        rear_slip = vehicle_model.rear_slip_

        h_wb = vehicle_model.wheelbase / 2
        h_w = track_width / 2

        if front_slip:
            self._add_slip_line(transform, self.last_transform, h_wb, -h_w)
            self._add_slip_line(transform, self.last_transform, h_wb, h_w)
        if rear_slip:
            self._add_slip_line(transform, self.last_transform, -h_wb, -h_w)
            self._add_slip_line(transform, self.last_transform, -h_wb, h_w)

        self.last_transform = transform

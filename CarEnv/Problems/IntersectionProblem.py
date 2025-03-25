import numpy as np

from .LaneletProblem import LaneletProblem


def traffic_driving_penalty(model, slip_start_deg=5.):
    vx, vy = model.v_rear_
    safe_vx = max(.1, abs(vx)) * (1. - 2. * (np.sign(vx) < 0.))
    rear_slip_angle = np.arctan2(vy, safe_vx)

    # Start punishing rear axle side slip angles above 5°
    abs_slip_deg = np.abs(rear_slip_angle) * 180 / np.pi
    penalty = max(0., abs_slip_deg - slip_start_deg) / 20.

    # Do not exceed upper bound
    penalty = min(1., penalty)
    return penalty


class LaneletBuilder:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.coords = [np.array([[x, y]])]

    def forward(self, distance):
        self.x += distance * np.cos(self.theta)
        self.y += distance * np.sin(self.theta)
        self.coords.append(np.array([[self.x, self.y]]))
        return self

    def curve90(self, radius):
        center_x = self.x - radius * np.sin(self.theta)
        center_y = self.y + radius * np.cos(self.theta)

        for ang in np.linspace(0., np.pi / 2 * np.sign(radius), 8)[1:]:
            self.coords.append(np.array([[center_x + radius * np.sin(ang + self.theta), center_y - radius * np.cos(ang + self.theta)]]))

        self.x, self.y = self.coords[-1][-1]
        self.theta = self.theta + np.pi / 2 * np.sign(radius)
        return self

    def build(self):
        return np.concatenate(self.coords)


def intersection_routes(lane_spacing=3., lane_length=50., right_turn_radius=4.5, left_turn_radius=4.5):
    result = {}

    s, l, lt, rt = lane_spacing, lane_length, left_turn_radius, right_turn_radius

    result["N->S"] = LaneletBuilder(-s / 2, -s / 2 - l, np.pi / 2).forward(s + 2 * l).build()
    result["N->W"] = LaneletBuilder(-s / 2, -s / 2 - l, np.pi / 2).forward(l - rt).curve90(rt).forward(l - rt).build()
    result["N->E"] = LaneletBuilder(-s / 2, -s / 2 - l, np.pi / 2).forward(l + s - lt).curve90(-lt).forward(l + s - lt).build()

    result["S->N"] = LaneletBuilder(s / 2, s / 2 + l, -np.pi / 2).forward(s + 2 * l).build()
    result["S->E"] = LaneletBuilder(s / 2, s / 2 + l, -np.pi / 2).forward(l - rt).curve90(rt).forward(l - rt).build()
    result["S->W"] = LaneletBuilder(s / 2, s / 2 + l, -np.pi / 2).forward(l + s - lt).curve90(-lt).forward(l + s - lt).build()

    result["W->E"] = LaneletBuilder(-s / 2 - l, s / 2, 0).forward(s + 2 * l).build()
    result["W->S"] = LaneletBuilder(-s / 2 - l, s / 2, 0).forward(l - rt).curve90(rt).forward(l - rt).build()
    result["W->N"] = LaneletBuilder(-s / 2 - l, s / 2, 0).forward(l + s - lt).curve90(-lt).forward(l + s - lt).build()

    result["E->W"] = LaneletBuilder(s / 2 + l, -s / 2, -np.pi).forward(s + 2 * l).build()
    result["E->N"] = LaneletBuilder(s / 2 + l, -s / 2, -np.pi).forward(l - rt).curve90(rt).forward(l - rt).build()
    result["E->S"] = LaneletBuilder(s / 2 + l, -s / 2, -np.pi).forward(l + s - lt).curve90(-lt).forward(l + s - lt).build()

    return result


class IntersectionProblem(LaneletProblem):
    """
    Single agent intersection navigation. The MARL variant is probably more interesting.
    """

    def __init__(self, env):
        super(IntersectionProblem, self).__init__(env, intersection_routes())

    def transform_ctx(self, ctx, width, height):
        from ..Rendering.BirdView import transform_for_pose

        transform_for_pose(ctx, width, height, self.env.ego_pose)

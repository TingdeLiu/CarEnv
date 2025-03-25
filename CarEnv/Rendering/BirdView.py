import cairo
import numpy as np
from .Rendering import BitmapRenderer, draw_old_car, draw_vehicle_state
from CarEnv.DeferredDraw import DeferredDraw
from typing import List, Optional, Union, Callable


class RenderCallback:
    def __call__(self, renderer, env, ctx):
        raise NotImplementedError


def transform_fixed_point(ctx: cairo.Context, width, height, point, scale=20.):
    x, y = point
    ctx.translate(width / 2, height / 2)
    ctx.scale(scale, scale)
    ctx.translate(-x, -y)


def transform_for_pose(ctx: cairo.Context, width, height, pose, scale=20., orient_forward=True, forward_center=.8):
    x, y, theta = pose

    if orient_forward:
        ctx.translate(width / 2, height * forward_center)
        ctx.rotate(-theta - np.pi / 2)
    else:
        ctx.translate(width / 2, height / 2)

    ctx.scale(scale, scale)
    ctx.translate(-x, -y)


def transform_for_bbox(ctx: cairo.Context, width, height, bbox):
    xmin, ymin, xmax, ymax = bbox

    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    scale = np.array([
        (cx - xmin) / (width / 2),
        (xmax - cx) / (width / 2),
        (cy - ymin) / (height / 2),
        (ymax - cy) / (height / 2),
    ])

    scale = 1. / np.max(scale)

    ctx.translate(width / 2, height / 2)
    ctx.scale(scale, scale)
    ctx.translate(-cx, -cy)


class BirdViewRenderer:
    def __init__(self, width, height, draw_physics=False, draw_sensors=False, hud=True, draw_grid=False,
                 callbacks: Optional[List[Union[Callable, RenderCallback]]] = None):
        if draw_grid:
            from warnings import warn
            warn("draw_grid is currently unsupported and will be ignored")

        self._bvr = BitmapRenderer(width, height)
        self._bvr.open()
        self.width = width
        self.height = height
        self.draw_physics = draw_physics
        self.draw_sensors = draw_sensors
        self.callbacks = callbacks or []
        self.draw_hud = hud
        self.draw_grid = draw_grid
        self.start_lights = None
        self.ghosts = None
        self.show_track = True
        self.show_digital_tachometer = False

    def _render_ctx(self, env, ctx):
        from .Gauge import Gauge

        ctx.identity_matrix()
        env.problem.transform_ctx(ctx, self.width, self.height)

        dd = DeferredDraw()

        env.problem.draw(dd)

        for obj in env.objects.values():
            if hasattr(obj, 'draw'):
                obj.draw(dd)

        dd.perform(ctx)

        self.render_ghosts(ctx, env)

        for cb in self.callbacks:
            cb(self, env, ctx)

        if self.draw_hud:
            if self.draw_sensors:
                for ag in env.agents:
                    agent = env.get_agent(ag)
                    for s_k, s_v in agent.sensors.items():
                        s_v.draw(ctx)

            if self.draw_physics:
                ctx.identity_matrix()
                ctx.translate(100, self.height - 100)
                draw_vehicle_state(ctx, env)

            ctx.identity_matrix()

            # Only render this if not MARL
            if hasattr(env, 'vehicle_model'):
                ctx.translate(self.width - 80, self.height - 80)
                Gauge(0, 100).draw(ctx, abs(env.vehicle_model.velocity[0]) * 3.6)
                self.render_pedals(ctx, env)

                if self.show_digital_tachometer:
                    self.render_digital_tachometer(ctx, env)

                env.action.render(ctx, self.width, self.height)

            self.render_start_lights(ctx)

    def render(self, env):
        ctx = self._bvr.clear()
        self._render_ctx(env, ctx)
        return self._bvr.get_data()

    def render_pdf(self, env, path):
        import cairo

        surface = cairo.PDFSurface(path, self.width, self.height)

        try:
            ctx = cairo.Context(surface)
            ctx.set_source_rgb(.8, .8, .8)
            ctx.fill()
            self._render_ctx(env, ctx)
        finally:
            surface.finish()

    def reset(self):
        self.ghosts = None

    def render_pedals(self, ctx: cairo.Context, env):
        from .Rendering import stroke_fill

        if not hasattr(env.action, 'throttle_position_'):
            return

        ctx.identity_matrix()
        ctx.translate(self.width - 200, self.height - 120)

        ctx.rectangle(40, 0, 20, 80)
        stroke_fill(ctx, (0., 0., 0.), (1., 1., 1.))

        bar_size = 80 * env.action.throttle_position_
        ctx.rectangle(40, 80 - bar_size, 20, bar_size)
        stroke_fill(ctx, (0., 0., 0.), (0., 1., 0.))

        ctx.rectangle(0, 0, 20, 80)
        stroke_fill(ctx, (0., 0., 0.), (1., 1., 1.))

        bar_size = 80 * env.action.brake_position_
        ctx.rectangle(0, 80 - bar_size, 20, bar_size)
        stroke_fill(ctx, (0., 0., 0.), (1., 0., 0.))

    def render_digital_tachometer(self, ctx: cairo.Context, env):
        from .Rendering import stroke_fill
        v = f"{int(env.vehicle_model.velocity[0] * 3.6)}"
        ctx.select_font_face("Latin Modern Mono", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
        advance = ctx.text_extents("0").x_advance
        height = ctx.text_extents("0").height
        ctx.set_font_size(100)
        ctx.identity_matrix()

        ctx.move_to(self.width / 2 - 250, self.height)
        ctx.line_to(self.width / 2 - 130, self.height - 120)
        ctx.line_to(self.width / 2 + 130, self.height - 120)
        ctx.line_to(self.width / 2 + 250, self.height)
        ctx.close_path()
        stroke_fill(ctx, (0., 0., 0.), (0.3, 0.3, 0.3))

        for i, k in enumerate(v.rjust(3, ' ')):
            ctx.move_to(self.width / 2 - advance * 3 / 2 + i * advance + 30, self.height - 60 + height / 2)
            ctx.text_path(k)
            stroke_fill(ctx, (0., 0., 0.), (1., .3, .3))

        if hasattr(env.vehicle_model, "front_slip_") and env.vehicle_model.front_slip_ is not None and \
                env.vehicle_model.front_slip_[0]:
            ctx.translate(0, self.height - 60 + height / 2)
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            ctx.arc(self.width / 2 - 100, -30, 20, 0, 2 * np.pi)
            ctx.new_sub_path()
            ctx.arc(self.width / 2 - 100, -30, 10, 0, 2 * np.pi)
            ctx.new_sub_path()
            ctx.rectangle(self.width / 2 - 120, -10, 40, 10)
            stroke_fill(ctx, (0., 0., 0.), (1., .3, .3))

    def render_ghosts(self, ctx: cairo.Context, env):
        if self.ghosts is None:
            return

        for color, pose in self.ghosts:
            draw_old_car(ctx, *pose, env.vehicle_model.wheelbase, 0., *env.collision_bb, braking=False,
                         color=color)

    def render_start_lights(self, ctx: cairo.Context):
        from .Rendering import stroke_fill
        r = 20

        if self.start_lights is None:
            return

        def draw(on):
            ctx.rectangle(-1.5 * r, -4 * r, 3 * r, 8 * r)
            stroke_fill(ctx, (0., 0., 0.), (.2, .2, .2))

            ctx.arc(0., 0., r, 0., 2 * np.pi)
            ctx.new_sub_path()
            ctx.arc(0., -2.5 * r, r, 0., 2 * np.pi)
            ctx.new_sub_path()
            ctx.arc(0., 2.5 * r, r, 0., 2 * np.pi)
            stroke_fill(ctx, (0., 0., 0.), (1., 0., 0.) if on else (.4, .2, .2))

        ctx.identity_matrix()
        ctx.translate(self.width * .5 - 3.5 * r, self.height * .2)
        draw(self.start_lights >= 1)
        ctx.identity_matrix()
        ctx.translate(self.width * .5 + 0.0 * r, self.height * .2)
        draw(self.start_lights >= 2)
        ctx.identity_matrix()
        ctx.translate(self.width * .5 + 3.5 * r, self.height * .2)
        draw(self.start_lights >= 3)

    def close(self):
        self._bvr.close()

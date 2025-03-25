from .DeferredDraw import DeferredDraw
from .Object import Object


class TireSlipObject(Object):
    def __init__(self, max_lines=500):
        self.max_lines = max_lines
        self.slip_lines = []

    def draw(self, draw: DeferredDraw):
        from .Rendering.DrawLevels import LEVEL_TIRE_MARKS
        draw.defer(LEVEL_TIRE_MARKS, self._deferred_draw)

    def _deferred_draw(self, ctx):
        import cairo

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for p1, p2 in self.slip_lines:
            ctx.move_to(*p1)
            ctx.line_to(*p2)

        ctx.set_source_rgb(0., 0., 0.)
        ctx.set_line_width(.15)  # Half the wheel width from car rendering
        ctx.stroke()

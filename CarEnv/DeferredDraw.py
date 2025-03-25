from bisect import bisect
from typing import Callable, Union
import cairo


class DeferredDraw:
    def __init__(self):
        # bisect() supports key since python 3.10, work around for earlier versions
        self._deferred_levels = []
        self._deferred_fns = []

    def defer(self, level: Union[int, float], fn: Callable[[cairo.Context], None]):
        idx = bisect(self._deferred_levels, level)
        self._deferred_levels.insert(idx, level)
        self._deferred_fns.insert(idx, fn)

    def perform(self, ctx: cairo.Context):
        for fn in self._deferred_fns:
            fn(ctx)

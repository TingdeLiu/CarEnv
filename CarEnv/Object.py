from .DeferredDraw import DeferredDraw


class Object:
    def update(self, env):
        pass

    def draw(self, draw: DeferredDraw):
        pass

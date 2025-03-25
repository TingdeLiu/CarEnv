def make_env_renderer(env, render_mode, render_kwargs):
    if render_mode is None:
        return NoRenderEnvRenderer()
    else:
        return EnvRenderer(env, render_mode, render_kwargs)


class AbstractEnvRenderer:
    def render_manual(self):
        raise NotImplementedError

    def render_automatic(self):
        raise NotImplementedError

    def reset(self):
        pass


class NoRenderEnvRenderer(AbstractEnvRenderer):
    def render_manual(self):
        return None

    def render_automatic(self):
        return None


class EnvRenderer(AbstractEnvRenderer):
    def __init__(self, env, render_mode, render_kwargs):
        # Set up rendering
        self.env = env
        self.__renderer = None
        self.__screen = None
        self.render_mode = render_mode
        render_kwargs = render_kwargs or {}
        self.__render_width, self.__render_height = render_kwargs.pop('width', 1280), render_kwargs.pop('height', 720)

        if self.render_mode == 'human':
            import pygame
            pygame.init()
            fs = render_kwargs.pop('fullscreen', False)
            screen_flags = pygame.FULLSCREEN if fs else 0
            rw, rh = (0, 0) if fs else (self.__render_width, self.__render_height)
            self.__screen = pygame.display.set_mode([rw, rh], flags=screen_flags, vsync=0)
            pygame.display.set_caption('CarEnv')
            self.__render_width, self.__render_height = self.__screen.get_width(), self.__screen.get_height()

        if self.render_mode in ['human', 'rgb_array']:
            from .BirdView import BirdViewRenderer
            kwargs = render_kwargs.pop('hints', {})

            # TODO: Should the renderer be cleaned up somewhere?
            self.__renderer = BirdViewRenderer(self.__render_width, self.__render_height, **kwargs)
        else:
            raise ValueError(f"{self.render_mode = }")

        if render_kwargs:
            raise ValueError(f"Unsupported options remain in render_kwargs: {list(render_kwargs.keys())}")

    def _render_impl(self, mode):
        import numpy as np

        if mode in ['human', 'rgb_array']:
            rgb_array = self.__renderer.render(self.env)

            if mode == 'rgb_array':
                return rgb_array

            import pygame
            # Consume events
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt()

            # pygame expects column major
            pygame.surfarray.blit_array(self.__screen, np.transpose(rgb_array, (1, 0, 2)))
            pygame.display.flip()
        else:
            raise ValueError(f"{mode = }")

    def render_manual(self):
        return self._render_impl('rgb_array')

    def render_automatic(self):
        if self.render_mode == 'human':
            return self._render_impl('human')

    def reset(self):
        self.__renderer.reset()

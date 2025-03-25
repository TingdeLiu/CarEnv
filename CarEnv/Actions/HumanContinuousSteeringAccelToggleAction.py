from typing import Tuple, Any

import cairo
import gymnasium as gym
import numpy as np

from .Action import Action


class HumanContinuousSteeringAccelToggleAction(Action):
    # TODO: If the vehicle model updates more frequently than the action value is read the vehicles speed may oscillate
    # TODO: when stopped and brakes are held. This is a problem caused by the joint control of brake and engine on the
    # TODO: kinematic vehicle model and cannot be fixed by the action.

    def __init__(self, env):
        super(HumanContinuousSteeringAccelToggleAction, self).__init__(env)

        import pygame

        self.env = env

        self.forward = True
        self.gear_switch_pressed = False

        pygame.init()

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
        else:
            self.joystick = None

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(2,))

    def render(self, ctx: cairo.Context, width, height):
        from ..Rendering.Rendering import stroke_fill

        ctx.save()
        ctx.identity_matrix()

        ctx.select_font_face("Latin Modern Mono", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
        ctx.set_font_size(100)
        f_advance = ctx.text_extents("0").x_advance

        ctx.move_to(30, height - 50)
        ctx.text_path("R")
        stroke_fill(ctx, (.0, .0, .0), (.3, .3, .3) if self.forward else (56 / 255, 103 / 255, 214 / 255))
        ctx.move_to(40 + f_advance, height - 50)
        ctx.text_path("D")
        stroke_fill(ctx, (.0, .0, .0), (56 / 255, 103 / 255, 214 / 255) if self.forward else (.3, .3, .3))

        ctx.restore()

    def interpret(self, act) -> Tuple[Any, Any]:
        import pygame
        from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN, K_SPACE

        # Ignore given action and read keyboard

        if self.joystick is not None:
            s = np.clip(self.joystick.get_axis(0) / .25, -1, 1)

            throttle_position = 1 - (self.joystick.get_axis(2) + 1) / 2
            brake_position = 1 - (self.joystick.get_axis(3) + 1) / 2

            change_gear = self.joystick.get_button(4)

            # print(f"{throttle_position = }, {brake_position = }")

            t = throttle_position - brake_position
        else:
            pressed = pygame.key.get_pressed()

            s = 0.0
            t = 0.0

            if pressed[K_LEFT]:
                s -= 1.0
            if pressed[K_RIGHT]:
                s += 1.0
            if pressed[K_UP]:
                t += 1.0
            if pressed[K_DOWN]:
                t -= 1.0

            change_gear = pressed[K_SPACE]

        if change_gear and not self.gear_switch_pressed:
            self.forward = not self.forward
        self.gear_switch_pressed = change_gear

        v_long, v_lat = self.env.vehicle_model.velocity

        if self.forward:
            if v_long < 0.:
                t = abs(t)
        else:
            t = -t
            if v_long > 0.:
                t = -abs(t)

        return s, np.array([t])

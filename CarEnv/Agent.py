from .Physics.AbstractCarModel import AbstractCarModel
from .Object import Object
from .Sensor import Sensor
from typing import Any, Dict


class Agent:
    def __init__(self, proxy: Object):
        """
        Initialize a new agent

        :param proxy: The proxy object of this agent in the scene
        """
        self.proxy: Object = proxy
        self.sensors: Dict[str, Sensor] = {}
        self.problem_data: Dict[str, Any] = {}
        self._pending_reward = 0
        self._pending_termination = False
        self._pending_truncation = False
        self._pending_info = {}

    def add_to_reward(self, val):
        self._pending_reward += val

    def add_info(self, key, val):
        self._pending_info[key] = val

    def set_reward(self, val):
        self._pending_reward = val

    def set_terminated(self, val):
        self._pending_termination = val

    def set_truncated(self, val):
        # For (mostly needless) compatibility
        self._pending_info['TimeLimit.truncated'] = val

        self._pending_truncation = val

    def get_and_clear_pending(self):
        result = self._pending_reward, self._pending_termination, self._pending_truncation, self._pending_info
        self._pending_reward = 0.
        self._pending_termination = False
        self._pending_truncation = False
        self._pending_info = {}
        return result

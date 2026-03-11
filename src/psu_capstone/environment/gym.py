from typing import Any

import gymnasium as g

from psu_capstone.environment.env import Environment


class Gym(g.Env):

    def __init__(self, observation_space: tuple[Any, ...], action_space: tuple[Any, ...]) -> None:
        super().__init__()

        self._env = Environment(observation_space, action_space)

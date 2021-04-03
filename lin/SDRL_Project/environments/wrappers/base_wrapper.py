from abc import ABC, abstractmethod


class BaseWrapper(ABC):
    def __init__(self):
        self.last_rgb_obs = None

    @abstractmethod
    def set_last_rgb_obs(self, rgb_obs):
        pass

    @abstractmethod
    def get_last_rgb_obs(self):
        pass

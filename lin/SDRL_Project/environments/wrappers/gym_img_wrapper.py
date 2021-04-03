import cv2
import gym
import numpy as np

from environments.wrappers.base_wrapper import BaseWrapper


class GymImageWrapper(gym.ObservationWrapper, BaseWrapper):

    def __init__(self, environment, screen_size=84, is_normalize=True):
        """Constructor for an Atari 2600 preprocessor.
        Args:
          environment: Gym environment whose observations are preprocessed.
          screen_size: int

        Raises:
          ValueError: if frame_skip or screen_size are not strictly positive.
        """
        gym.Wrapper.__init__(self, environment)
        BaseWrapper.__init__(self)
        self.environment = environment
        self.screen_size = screen_size
        self.is_normalize = is_normalize
        self.original_shape = self.observation_space.shape

        # buffer to save grayscale image
        obs_dims = self.environment.observation_space
        self.screen_buffer = np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)

        high = 1.0 if is_normalize else 255
        self.observation_space = gym.spaces.Box(low=0, high=high, shape=(screen_size, screen_size))

    def observation(self, observation):
        self.set_last_rgb_obs(observation)

        if hasattr(self.environment, 'ale'):
            # fetch grayscale image directly
            self.environment.ale.getScreenGrayscale(self.screen_buffer)
        else:
            obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            np.copyto(self.screen_buffer, obs)

        transformed_image = cv2.resize(self.screen_buffer,
                                       (self.screen_size, self.screen_size),
                                       interpolation=cv2.INTER_AREA)
        if self.is_normalize:
            transformed_image = transformed_image.astype(np.float32)
            transformed_image = transformed_image / 255.0
        return transformed_image

    def set_last_rgb_obs(self, rgb_obs):
        self.last_rgb_obs = np.copy(rgb_obs)

    def get_last_rgb_obs(self):
        return self.last_rgb_obs

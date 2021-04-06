# Modifications Copyright (c) 2018 Uber Technologies, Inc.
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random

import gym
from gym.spaces.box import Box
import numpy as np
import cv2

from environments.wrappers.base_wrapper import BaseWrapper


class DopamineAtariPreprocessor(gym.Wrapper, BaseWrapper):
    """A class implementing image preprocessing for Atari 2600 agents.

    Specifically, this provides the following subset from the JAIR paper
    (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

      * Frame skipping (defaults to 4).
      * Terminal signal when a life is lost (off by default).
      * Grayscale and max-pooling of the last two frames.
      * Downsample the screen to a square image (defaults to 84x84).

    More generally, this class follows the preprocessing guidelines set down in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    """

    def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
                 screen_size=84, is_normalize=True, is_stochastic=False):
        """Constructor for an Atari 2600 preprocessor.
        Args:
          environment: Gym environment whose observations are preprocessed.
          frame_skip: int, the frequency at which the agent experiences the game.
          terminal_on_life_loss: bool, If True, the step() method returns
            is_terminal=True whenever a life is lost. See Mnih et al. 2015.
          screen_size: int, size of a resized Atari 2600 frame.

        Raises:
          ValueError: if frame_skip or screen_size are not strictly positive.
        """
        gym.Wrapper.__init__(self, environment)
        BaseWrapper.__init__(self)

        self.environment = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.is_normalize = is_normalize

        obs_dims = self.environment.observation_space
        if not self.is_normalize:
            self.observation_space = gym.spaces.Box(low=0
                                                    , high=255
                                                    , shape=(self.screen_size, self.screen_size))
        else:
            self.observation_space = gym.spaces.Box(low=0
                                                    , high=1
                                                    , shape=(self.screen_size, self.screen_size))

        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]
        self.game_over = False
        self.lives = 0  # Will need to be set by reset().
        self.is_stochastic = is_stochastic

    def reset(self):
        """Resets the environment.

        Returns:
            observation: numpy array, the initial observation emitted by the environment.
        """
        self.environment.reset()
        self.lives = self.environment.ale.lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()

    def step(self, action):
        """Applies the given action in the environment.

        Remarks:

        * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
        * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.

        Args:
            action: The action to be executed.

        Returns:
            observation: numpy array, the observation following the action.
            reward: float, the reward following the action.
            is_terminal: bool, whether the environment has reached a terminal state.
              This is true when a life is lost and terminal_on_life_loss, or when the
              episode is over.
            info: Gym API's info data structure.
        """
        accumulated_reward = 0.
        frame_skip = self.frame_skip if not self.is_stochastic else random.randint(2, self.frame_skip)

        for time_step in range(frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, game_over, info = self.environment.step(action)
            accumulated_reward += reward

            if self.terminal_on_life_loss:
                new_lives = self.environment.ale.lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            if is_terminal:
                break
            # We max-pool over the last two frames, in grayscale.
            elif time_step >= frame_skip - 2:
                t = time_step - (frame_skip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

        # Pool the last two observations.
        observation = self._pool_and_resize()
        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.

        The returned observation is stored in 'output'.

        Args:
            output: numpy array, screen buffer to hold the returned observation.

        Returns:
            observation: numpy array, the current observation in grayscale.
        """
        self.environment.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

        For efficiency, the transformation is done in-place in self.screen_buffer.

        Returns:
        transformed_screen: numpy array, pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                       out=self.screen_buffer[0])

        transformed_image = cv2.resize(self.screen_buffer[0],
                                       (self.screen_size, self.screen_size),
                                       interpolation=cv2.INTER_AREA)
        img = np.asarray(transformed_image, dtype=self.screen_buffer[0].dtype)
        if self.is_normalize:
            img = img.astype(np.float32)
            img = img / 255.0
        return img

    def get_last_rgb_obs(self):
        return self.environment.ale.get_screen_rgb()

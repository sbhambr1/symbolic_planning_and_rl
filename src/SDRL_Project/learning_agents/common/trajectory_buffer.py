from learning_agents.utils.tensor_utils import get_device
from learning_agents.utils.trajectory_utils import get_flatten_indexed_trajs, TRAJECTORY_INDEX
import numpy as np
import torch


device = get_device()


class TrajectoryBuffer:
    """
    Rather than using transition as basic unit, TrajectoryBuffer use individual traj as basic unit, format: List[dict]
    Attributes:
        traj_buffer (list of dict): a list (buffer) of list (trajectories, each trajectory is a dict)
        buffer_size (int): size of buffers
        batch_size (int): batch size for training
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(self, buffer_size, batch_size=32, trajectories=None):
        """
        Initialize a ReplayBuffer object.
        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
        """
        assert 0 < batch_size <= buffer_size

        self.traj_buffer = None

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.length = 0
        self.idx = 0

        self._initialize_buffers()

    def _initialize_buffers(self):
        """
        Initialize buffers
        """
        self.traj_buffer = [None for _ in range(self.buffer_size)]

    def __len__(self):
        """Return the current size of internal memory."""
        return self.length

    def add(self, indexed_traj):
        """
        Add a new experience to memory.
        """
        self.traj_buffer[self.idx] = indexed_traj

        self.idx += 1
        self.idx = 0 if self.idx % self.buffer_size == 0 else self.idx
        self.length = min(self.length + 1, self.buffer_size)

    def extend(self, trajectories):
        """
        Add experiences to memory.
        """
        for _, traj in enumerate(trajectories):
            self.add(traj)

    def sample(self, indices=None, batch_size=None):
        """
        Randomly sample a batch of experiences from memory.
        indices: List[int] = None)
        :return: (list of trajectories, dict of related information)
        """
        if batch_size is None and indices is None:
            batch_size = self.batch_size
            assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=batch_size, replace=False)

        sampled_trajs = [self.traj_buffer[i] for i in indices]

        return sampled_trajs

    def get_experiences_from_trajs(self, indices=None):
        if indices is not None:
            sampled_trajs = [self.traj_buffer[idx] for idx in indices]
        else:
            sampled_trajs = self.traj_buffer[:self.length]

        flatten_trajs = get_flatten_indexed_trajs(sampled_trajs)
        states = np.array(flatten_trajs[TRAJECTORY_INDEX.STATE.value])
        actions = np.array(flatten_trajs[TRAJECTORY_INDEX.ACTION.value])
        rewards = np.array(flatten_trajs[TRAJECTORY_INDEX.REWARD.value])
        next_states = np.array(flatten_trajs[TRAJECTORY_INDEX.NEXT_STATE.value])
        dones = np.array(flatten_trajs[TRAJECTORY_INDEX.DONE.value])
        infos = np.array(flatten_trajs[TRAJECTORY_INDEX.INFO.value])

        return states, actions, rewards, next_states, dones, infos






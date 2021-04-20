# epsilon greedy strategy
# --------------------------------

class Epsilon_Greedy:
    def __init__(self, decay, min_eps, max_eps, random_step=0, mode='exponential'):
        """
        Epsilon greedy strategy with two modes available:
            Mode 'exponential': exponentially decay each episode
            Mode 'linear-step': linear decay each step
            Mode 'linear-episode': linear decay each episode
        """
        self.decay = decay
        self.mode = mode
        self.random_step = random_step
        self.min_eps = min_eps
        self.max_eps = max_eps
        # member variables
        self.starting_episode = -1

    def get_epsilon(self, i_step, i_episode):
        if i_step < self.random_step:
            return 1.0
        if i_step >= self.random_step and self.starting_episode < 0:
            self.starting_episode = i_episode

        if self.mode == 'exponential-episode':
            return max(self.min_eps, self.max_eps * (self.decay ** (i_episode - self.starting_episode)))
        elif self.mode == 'linear-step':
            return max(self.min_eps,
                       self.max_eps + (self.min_eps - self.max_eps) * float(i_step - self.random_step) / self.decay)
        elif self.mode == 'linear-episode':
            return max(self.min_eps,
                       self.max_eps + (self.min_eps - self.max_eps) * float(
                           i_episode - self.starting_episode) / self.decay)
        else:
            return 0

# Exploration strategy factory function
# -----------------------------------------

def get_strategy(name, config):
    if name == 'epsilon-greedy':
        from learning_agents.exploration.epsilon_greedy import Epsilon_Greedy
        return Epsilon_Greedy(decay=config.epsilon_decay, min_eps=config.min_epsilon, max_eps=config.max_epsilon,
                              random_step=config.init_random_actions, mode=config.epsilon_strategy)
    else:
        return None

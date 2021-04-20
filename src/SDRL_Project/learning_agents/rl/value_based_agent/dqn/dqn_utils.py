import torch
import torch.nn.functional as F

from learning_agents.utils.tensor_utils import get_device

device = get_device()

LOG_REG = 1e-8


def calculate_dqn_loss(model, target_model, experiences, gamma,
                       use_double_q_update=False, reward_clip=None, reward_scale=None):
    """Return element-wise dqn loss and Q-values."""
    states, actions, rewards, next_states, dones = experiences[:5]
    if reward_scale is not None:
        rewards = rewards * reward_scale
    if reward_clip is not None:
        rewards = torch.clamp(rewards, min=reward_clip[0], max=reward_clip[1])

    # compute current values
    q_values = model(states)

    # According to noisynet paper,
    # it re-samples noisynet parameters on online network when using double q
    # but we don't because there is no remarkable difference in performance.
    next_q_values = model(next_states)
    curr_q_value = q_values.gather(1, actions.long())

    # compute target values
    next_target_q_values = target_model(next_states)
    if use_double_q_update:
        # Double DQN
        next_q_value = next_target_q_values.gather(
            1, next_q_values.argmax(1).unsqueeze(1)
        )
    else:
        # Ordinary DQN
        next_q_value = next_target_q_values.gather(
            1, next_target_q_values.argmax(1).unsqueeze(1)
        )

    # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
    #       = r                       otherwise
    masks = 1 - dones
    target = rewards + gamma * next_q_value * masks
    target = target.to(device)

    # calculate dq loss
    dq_loss_element_wise = F.mse_loss(curr_q_value, target.detach(), reduction="none")

    return dq_loss_element_wise, q_values





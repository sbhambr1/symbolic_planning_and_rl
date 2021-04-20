# train value-based learning agent, e.g. q-learning
# --------------------------------------------------
import time
from collections import deque
import numpy as np

from learning_agents.common.eval_agent import eval_agent


class Train_Value_Based_Agent:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.old_life = -1

    def step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        done = (
            True if self.agent.episode_step >= self.agent.args.max_traj_len - 1 else done
        )
        if 'ale.lives' in info and self.agent.hyper_params.terminate_life_loss:
            current_lives = info['ale.lives']
            if current_lives > self.old_life:
                self.old_life = current_lives
            elif current_lives < self.old_life:
                done = True
        return next_state, reward, done, info

    def train(self):
        # whether do pretraining
        if hasattr(self.agent, 'pretrain'):
            self.agent.pretrain()

        avg_scores_window = deque(maxlen=self.agent.args.avg_score_window)
        eval_scores_window = deque(maxlen=self.agent.args.eval_score_window)

        while self.agent.total_step < self.agent.args.max_step:
            self.agent.i_episode += 1
            # check whether to perform evaluation rollout
            self.agent.testing = (self.agent.i_episode % self.agent.args.eval_period == 0)
            # init episode
            state = self.env.reset()
            self.agent.episode_step = 0
            losses = list()
            done = False
            score = 0
            t_begin = time.time()

            while not done:
                if self.agent.args.render \
                        and self.agent.i_episode >= self.agent.args.render_after \
                        and self.agent.i_episode % self.agent.args.render_freq == 0:
                    self.env.render()

                action = self.agent.select_action(state)
                next_state, reward, done, info = self.step(action)
                score += reward

                # save the new transition
                transition = (state, action, reward, next_state, done, info)
                self.agent.add_transition_to_memory(transition)
                state = next_state

                # update agent variables
                self.agent.total_step += 1
                self.agent.episode_step += 1
                if self.agent.total_step % self.agent.args.save_period == 0:
                    self.agent.save_params(self.agent.total_step)

                # train the agent
                if len(self.agent.memory) >= self.agent.hyper_params.update_starts_from:
                    if self.agent.total_step % self.agent.hyper_params.train_freq == 0:
                        for _ in range(self.agent.hyper_params.policy_multiple_update):
                            loss = self.agent.update_model()
                            losses.append(loss)  # for logging

            # save score info
            avg_scores_window.append(score)
            # post-episode update
            self.agent.do_post_episode_update()
            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.agent.episode_step

            # log eval rollout info
            if self.agent.testing:
                eval_scores_window.append(score)
                # noinspection PyStringFormat
                eval_log_info = {
                    'eval score': score,
                    "eval window avg": np.mean(eval_scores_window),
                }
                print('[EVAL INFO] episode: %d, total step %d, %s\n'
                      % (self.agent.i_episode, self.agent.total_step, str(eval_log_info)))

                if self.agent.logger is not None:
                    self.agent.logger.log_wandb(eval_log_info, step=self.agent.total_step)
            # log general training info
            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (avg_loss, score, avg_time_cost, np.mean(avg_scores_window))
                self.agent.write_log(log_value)

        # termination
        self.env.close()
        self.agent.save_params(self.agent.total_step)

    def test(self, n_episode=5, verbose=True):
        from utils.info_displayer import InfoDisplayer
        info_displayer = InfoDisplayer(screen_height=150 * 6, screen_width=250, frame_time=0.1)
        eval_agent(self.env, self.agent, n_episode, verbose=verbose, render=self.agent.args.render,
                   info_displayer=info_displayer)
        # termination
        self.env.close()

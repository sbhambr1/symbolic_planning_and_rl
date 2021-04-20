def eval_agent(env, agent, n_episode, render=False, max_step=None, info_displayer=None, verbose=False):
    def step(a, current_step):
        next_s, r, d, env_info = env.step(a)
        d = (
            True if max_step and current_step == max_step else d
        )
        return next_s, r, d, env_info

    if hasattr(agent, 'testing'):
        agent_testing = agent.testing   # save current agent testing status
        agent.testing = True
    # save scores
    scores = []

    for i_episode in range(int(n_episode)):
        state = env.reset()
        done = False
        score = 0
        episode_step = 0

        while not done:
            if render:
                if info_displayer is not None:
                    info_displayer.display_img_rgb(
                        rgb_img=state if 'get_last_rgb_obs' not in dir(env) else env.get_last_rgb_obs())
                    info_displayer.refresh()
                else:
                    env.render()

            action = agent.select_action(state)
            next_state, reward, done, _ = step(a=action, current_step=episode_step)
            state = next_state
            score += reward
            if reward != 0 and verbose:
                print('[TEST] reward: ', reward, ', score: ', score, ', step: ', episode_step)
            episode_step += 1

        scores.append(score)
        if verbose:
            print('[TEST] Episode: {0}, Score: {1}, Episode Steps: {2}\n'.format(i_episode, score, episode_step))

    if hasattr(agent, 'testing'):
        # noinspection PyUnboundLocalVariable
        agent.testing = agent_testing

    return scores

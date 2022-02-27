import numpy as np
import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

from babyai.evaluate import ManyEnvs

from wrappers import AmbiguousInstructionsWrapper, NonsenseInstructionsWrapper


# Returns the performance of the agent on the environment for a particular number of episodes.
def batch_evaluate(agent, env_name, seed, episodes, return_obss_actions=False, pixel=False, nonsense=False, ambiguous=False, prob_ambiguous=0.5):
    assert not (nonsense and ambiguous), f"Nonsense and Ambiguous are both true"

    num_envs = min(256, episodes)

    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)

        if ambiguous:
            env = AmbiguousInstructionsWrapper(env, prob_ambiguous=prob_ambiguous)
        elif nonsense:
            env = NonsenseInstructionsWrapper(env)


        if pixel:
            env = RGBImgPartialObsWrapper(env)
        envs.append(env)
    env = ManyEnvs(envs)

    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "observations_per_episode": [],
        "actions_per_episode": [],
        "seed_per_episode": []
    }

    for i in range((episodes + num_envs - 1) // num_envs):
        seeds = range(seed + i * num_envs, seed + (i + 1) * num_envs)
        env.seed(seeds)

        many_obs = env.reset()

        cur_num_frames = 0
        num_frames = np.zeros((num_envs,), dtype='int64')
        returns = np.zeros((num_envs,))
        already_done = np.zeros((num_envs,), dtype='bool')
        if return_obss_actions:
            obss = [[] for _ in range(num_envs)]
            actions = [[] for _ in range(num_envs)]
        while (num_frames == 0).any():
            action = agent.act_batch(many_obs)['action']
            if return_obss_actions:
                for i in range(num_envs):
                    if not already_done[i]:
                        obss[i].append(many_obs[i])
                        actions[i].append(action[i].item())
            many_obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            done = np.array(done)
            just_done = done & (~already_done)
            returns += reward * just_done
            cur_num_frames += 1
            num_frames[just_done] = cur_num_frames
            already_done[done] = True

        logs["num_frames_per_episode"].extend(list(num_frames))
        logs["return_per_episode"].extend(list(returns))
        logs["seed_per_episode"].extend(list(seeds))
        if return_obss_actions:
            logs["observations_per_episode"].extend(obss)
            logs["actions_per_episode"].extend(actions)

    return logs

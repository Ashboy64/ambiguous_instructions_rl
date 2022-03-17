import numpy as np
import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

from babyai.evaluate import ManyEnvs

from wrappers import AmbiguousInstructionsWrapper, NonsenseInstructionsWrapper


# from preprocess import make_sample
import torch
from copy import copy

from tqdm import tqdm, trange

def make_sample2(image, command):
    image = list(image.flatten().astype(str))
    sample = ' '.join(image) + ' ' + command
    return sample

def preprocess_image_and_mission(image, mission, tokenizer, vocab, pad_id, device):
    # sample = make_sample2(image, mission)
    # sample = tokenizer(sample)
    # # pad_id = "<PAD>"
    # while len(sample) < 156:
    #     sample.append(pad_id)
    # text = []
    # for word in sample:
    #     text.append(vocab[word])
    # text = torch.Tensor(text).long()
    # return text

    sample = make_sample2(image, mission)
    if vocab is not None:
        assert pad_id == "<PAD>"
        sample = tokenizer(sample)
    else:
        sample = tokenizer(sample)["input_ids"]

    while len(sample) < 156:
        sample.append(pad_id)

    if vocab is not None:
        text = []
        for word in sample:
            text.append(vocab[word])
        text = torch.Tensor(text).long()
    else:
        text = torch.Tensor(sample).long()

    text = text.to(device)

    return text

# Returns the performance of the agent on the environment for a particular number of episodes.
def batch_evaluate(agent, env_name, seed, episodes, return_obss_actions=False, pixel=False, nonsense=False, ambiguous=False, prob_ambiguous=0.5, classifier_network=None, vocab=None, tokenizer=None):
    print("Custom batch evaluate!!!")
    assert not (nonsense and ambiguous), f"Nonsense and Ambiguous are both true"

    if classifier_network is not None:
        device = next(classifier_network.parameters()).device


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
        "seed_per_episode": [],

        "rate_unambiguous_per_episode": [],
        "rate_ambiguous_per_episode": [],
        "rate_correct_pred_per_episode": [],
        "rate_tp_per_episode": [],
        "rate_tn_episode": [],
        "rate_fp_per_episode": [],
        "rate_fn_per_episode": [],

        "n_unambiguous_per_episode": [],
        "n_ambiguous_per_episode": [],
        "n_correct_pred_per_episode": [],
        "n_tp_per_episode": [],
        "n_tn_episode": [],
        "n_fp_per_episode": [],
        "n_fn_per_episode": [],
    }

    for i in range((episodes + num_envs - 1) // num_envs):
        seeds = range(seed + i * num_envs, seed + (i + 1) * num_envs)
        env.seed(seeds)


        many_obs = env.reset()

        if i == 0 and (ambiguous or nonsense):
            print("True mission:", env.envs[0].true_mission)
            print("Mission:", many_obs[0]["mission"])

        cur_num_frames = 0
        num_frames = np.zeros((num_envs,), dtype='int64')
        returns = np.zeros((num_envs,))
        already_done = np.zeros((num_envs,), dtype='bool')

        n_ambiguous = np.zeros((num_envs,), dtype='int64')
        n_unambiguous = np.zeros((num_envs,), dtype='int64')
        n_correct_pred = np.zeros((num_envs,), dtype='int64')
        n_tp = np.zeros((num_envs,), dtype='int64')
        n_tn = np.zeros((num_envs,), dtype='int64')
        n_fp = np.zeros((num_envs,), dtype='int64')
        n_fn = np.zeros((num_envs,), dtype='int64')

        if return_obss_actions:
            obss = [[] for _ in range(num_envs)]
            actions = [[] for _ in range(num_envs)]
        while (num_frames == 0).any():

            if ambiguous and classifier_network is not None:
                for env_idx in range(len(many_obs)):
                    # mission = many_obs[env_idx]["mission"]
                    # image = many_obs[env_idx]["image"]
                    # true_mission = env.envs[env_idx].true_mission

                    text = preprocess_image_and_mission(many_obs[env_idx]["image"],
                                                        many_obs[env_idx]["mission"],
                                                        tokenizer,
                                                        vocab,
                                                        classifier_network.config.pad_token_id if vocab is None else "<PAD>",
                                                        device)

                    # print(f"[{env_idx}] mission: {mission}, env.envs[env_idx].mission: {env.envs[env_idx].mission}")

                    if not already_done[env_idx]:
                        assert many_obs[env_idx]["mission"] == env.envs[env_idx].mission, "[{}] mission: {}, env.envs[{}].mission: {}".format(env_idx, many_obs[env_idx]["mission"], env_idx, env.envs[env_idx].mission)

                    assert not classifier_network.training
                    if vocab is not None:
                        pred_prob_ambiguous = classifier_network(text.unsqueeze(0)).item()
                        pred_is_ambiguous = pred_prob_ambiguous > 0.5
                    else:
                        output = classifier_network(text.unsqueeze(0))
                        logits = output.logits.squeeze()
                        pred_is_ambiguous = (logits[1] > logits[0]).item()

                    if pred_is_ambiguous:
                        many_obs[env_idx]["mission"] = copy(env.envs[env_idx].true_mission)

                    if not already_done[env_idx]:
                        if env.envs[env_idx].is_currently_ambiguous:
                            n_ambiguous[env_idx] += 1
                            if pred_is_ambiguous:
                                n_tp[env_idx] += 1 # is ambiguous and predict ambiguous
                                n_correct_pred[env_idx] += 1 # correct prediction
                            else:
                                n_fn[env_idx] += 1 # is ambiguous and predict not ambiguous
                        else:
                            n_unambiguous[env_idx] += 1
                            if pred_is_ambiguous:
                                n_fp[env_idx] += 1 # is not ambiguous and predict ambiguous
                            else:
                                n_tn[env_idx] += 1 # is not ambiguous and predict not ambiguous
                                n_correct_pred[env_idx] += 1 # correct prediction

            action = agent.act_batch(many_obs)['action']
            # if return_obss_actions:
            #     for i in range(num_envs):
            #         if not already_done[i]:
            #             obss[i].append(many_obs[i])
            #             actions[i].append(action[i].item())
            if return_obss_actions:
                for j in range(num_envs):
                    if not already_done[j]:
                        obss[j].append(many_obs[j])
                        actions[j].append(action[j].item())
            many_obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            done = np.array(done)
            just_done = done & (~already_done)
            returns += reward * just_done
            cur_num_frames += 1
            num_frames[just_done] = cur_num_frames
            already_done[done] = True

            if classifier_network is not None:
                print(f"[{i + 1}/{(episodes + num_envs - 1) // num_envs}] [num_frames] min: {num_frames.min()} max: {num_frames.max()}, [cur_num_frames]: {cur_num_frames} device: {device}")

        logs["num_frames_per_episode"].extend(list(num_frames))
        logs["return_per_episode"].extend(list(returns))
        logs["seed_per_episode"].extend(list(seeds))
        if return_obss_actions:
            logs["observations_per_episode"].extend(obss)
            logs["actions_per_episode"].extend(actions)

        logs["rate_ambiguous_per_episode"].extend(list(n_ambiguous / num_frames))
        logs["rate_unambiguous_per_episode"].extend(list(n_unambiguous / num_frames))
        logs["rate_correct_pred_per_episode"].extend(list(n_correct_pred / num_frames))
        logs["rate_tp_per_episode"].extend(list(n_tp / num_frames))
        logs["rate_tn_episode"].extend(list(n_tn / num_frames))
        logs["rate_fp_per_episode"].extend(list(n_fp / num_frames))
        logs["rate_fn_per_episode"].extend(list(n_fn / num_frames))

        logs["n_ambiguous_per_episode"].extend(list(n_ambiguous))
        logs["n_unambiguous_per_episode"].extend(list(n_unambiguous))
        logs["n_correct_pred_per_episode"].extend(list(n_correct_pred))
        logs["n_tp_per_episode"].extend(list(n_tp ))
        logs["n_tn_episode"].extend(list(n_tn))
        logs["n_fp_per_episode"].extend(list(n_fp))
        logs["n_fn_per_episode"].extend(list(n_fn))

    return logs

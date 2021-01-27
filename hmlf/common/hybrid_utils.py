import numpy as np


def to_hybrid_discrete(actions, action_space):

    col, row = np.where(actions[:, :2] == 1)

    N = action_space[0].n

    a_low = [0] * N + sum([list(action_space[i].low * action_space[i].shape[0]) for i in range(1, len(action_space))], [])
    a_high = [1] * N + sum([list(action_space[i].high * action_space[i].shape[0]) for i in range(1, len(action_space))], [])
    actions = np.clip(actions, a_low, a_high)

    splits = [action_space[0].n] + [action_space[i].shape[0] for i in range(1, len(action_space))]
    splits = np.cumsum(splits)
    discrete_spl, *continuos_spl = np.split(actions, splits[:-1], axis=1)

    clipped_actions = [[] for i in range(actions.shape[0])]
    for i in range(actions.shape[0]):
        clipped_actions[i].append(row[i])
        for task_param in continuos_spl:
            clipped_actions[i].append(task_param[i])
    return clipped_actions

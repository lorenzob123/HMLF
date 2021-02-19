import numpy as np
from hmlf.spaces import Tuple


def onehot_hybrid_2_tuple_hybrid(actions: np.ndarray, action_space: Tuple) -> list: 
    """ This function takes in actions array for tuple action spaces and processes it to
        output the clipped_actions that can be used in the env step function 
         (as in env.step(clipped_actions))

    Args:
        actions (np.ndarray): an np.ndarray which has the actions with one-hot encoding for
                              the discrete task, concatenated with all the parameters for all
                              the tasks. 
        action_space (hmlf.spaces.Tuple): The action space of the env being used. It is expected to be
                                         of type Tuple with the first entry Discrete and the remaining
                                         ones Box.

    Returns:
        clipped_actions(list(list)): the actions formatted to be used in the environment's step function.
    """
    N = action_space[0].n

    # task is a vector with the task value for each environment
    _, task = np.where(actions[:, :N] == np.max(actions[:, :N], axis=0).squeeze())

    # We clip thee actions according to action_space[i].low and  action_space[i].high and
    # the 0 and 1 are for the discrete actions in the onehot encoding.
    a_low = [0] * N + sum([list(action_space[i].low * action_space[i].shape[0])
                           for i in range(1, len(action_space))], [])
    a_high = [1] * N + sum([list(action_space[i].high * action_space[i].shape[0])
                            for i in range(1, len(action_space))], [])
    actions = np.clip(actions, a_low, a_high)

    # We split actions to get the paramters for each task separeted
    idx_splits = [action_space[0].n] + [action_space[i].shape[0] for i in range(1, len(action_space))]
    idx_splits = np.cumsum(idx_splits)
    _, *continuos_spl = np.split(actions, idx_splits[:-1], axis=1)


    # We loop over all the enviroments and we format the actions as:
    # [Discrete(N), Box(param1_size), Box(param2_size), ...].
    clipped_actions = [[] for i in range(actions.shape[0])]
    for i in range(actions.shape[0]):
        clipped_actions[i].append(task[i])
        for task_param in continuos_spl:
            clipped_actions[i].append(task_param[i])
    return clipped_actions

from typing import Tuple

import numpy as np


class ExperimentHelpers(object):

    @staticmethod
    def filter_by_initial_state(data: Tuple[np.ndarray, np.ndarray], p: float, lower: float, upper: float):
        states, rewards = data
        n = states.shape[0]
        in_range = (lower < states[:, 0]) & (states[:, 0] < upper)
        probs = np.ones(n)
        probs[in_range] = p
        probs /= probs.sum()
        sampled_idxes = np.random.choice(np.arange(n), size=n, replace=True, p=probs)
        return states[sampled_idxes], rewards[sampled_idxes]

from typing import List

import numpy as np
from sepsis_sim import MDP


class SepsisTrajectories(object):

    n_actions = MDP.Action.NUM_ACTIONS_TOTAL
    n_states = MDP.State.NUM_FULL_STATES
    gamma = 0.9

    @staticmethod
    def get_state_idxes_by_type(non_absorbing=True):
        state_idxes = []
        for state_idx in range(MDP.State.NUM_FULL_STATES):
            state = MDP.State(state_idx, idx_type='full')
            if non_absorbing:
                if not state.check_absorbing_state():
                    state_idxes.append(state_idx)
            else:
                if state.check_absorbing_state():
                    state_idxes.append(state_idx)
        return state_idxes

    @staticmethod
    def get_cumulative_reward_from_rewards(rewards: List[float]):
        n = len(rewards)
        discount_rates = SepsisTrajectories.gamma ** np.arange(n)
        return np.multiply(np.array(rewards), discount_rates).sum() * (1 - SepsisTrajectories.gamma)

    @staticmethod
    def simulate_from_s0(s0_idx, action_idx, n_periods, actions_for_t, return_vec=False):
        dummy_pol = np.ones((MDP.State.NUM_OBS_STATES, SepsisTrajectories.n_actions)) / SepsisTrajectories.n_actions
        states = [MDP.State(s0_idx, idx_type='full').get_full_state_vector()] if return_vec else [s0_idx]
        rewards = []
        mdp = MDP.MDP(init_state_idx=s0_idx, policy_array=dummy_pol, p_diabetes=0., init_state_idx_type='full')
        for t in range(n_periods):
            r = mdp.transition(MDP.Action(action_idx=action_idx if t < actions_for_t else 0))
            rewards.append(r)
            states.append(mdp.state.get_full_state_vector() if return_vec else mdp.state.get_state_idx('full'))
            if mdp.state.check_absorbing_state():
                break

        return np.vstack(states), rewards

    @staticmethod
    def get_fake_transitions(obs_per_state):
        s0s_idxes = SepsisTrajectories.get_state_idxes_by_type(non_absorbing=False)
        s0s = [MDP.State(state_idx, idx_type='full').get_full_state_vector() for state_idx in s0s_idxes]
        s0s = np.vstack(np.array(s0s))
        states = np.dstack((s0s, s0s))
        rewards = np.zeros((len(s0s_idxes) * obs_per_state, 1))

        return np.tile(states, [obs_per_state, 1, 1]), rewards

    @staticmethod
    def get_transitions(action_idx, obs_per_state):
        s0s = SepsisTrajectories.get_state_idxes_by_type() * obs_per_state
        non_abs_transitions = np.zeros((len(s0s), 8, 2))
        non_abs_rewards = np.zeros((len(s0s), 1))
        for obs_idx, s0 in enumerate(s0s):
            states_from_s0, rewards_from_s0 = SepsisTrajectories.simulate_from_s0(
                s0, action_idx, 1, actions_for_t=np.inf, return_vec=True
            )
            non_abs_transitions[obs_idx, :, :] = states_from_s0.T
            non_abs_rewards[obs_idx, :] = rewards_from_s0

        abs_transitions, abs_rewards = SepsisTrajectories.get_fake_transitions(obs_per_state // 2)

        transitions = np.vstack((non_abs_transitions, abs_transitions))
        rewards = np.vstack((non_abs_rewards, abs_rewards))
        N = transitions.shape[0]
        actions = np.ones((N, 1)) if action_idx > 0 else np.zeros((N, 1))

        return transitions, actions, rewards

    @staticmethod
    def get_cumulative_reward_for_action(action_idx, n_periods, obs_per_state: int, actions_for_t: int):
        s0s = SepsisTrajectories.get_state_idxes_by_type() * obs_per_state
        rewards = []
        Ts = []
        for s0 in s0s:
            _, rewards_list = SepsisTrajectories.simulate_from_s0(s0, action_idx, n_periods, actions_for_t=actions_for_t)
            Ts.append(len(rewards_list))
            rewards.append(SepsisTrajectories.get_cumulative_reward_from_rewards(rewards_list))

        return np.mean(rewards)

    @staticmethod
    def get_true_ates(n_periods=120, obs_per_state=20, actions_for_t=np.inf):
        treatment_effects = {}
        for action_idx in range(1, SepsisTrajectories.n_actions):
            treatment_effects[action_idx] = SepsisTrajectories.get_true_ate(
                action_idx, n_periods, obs_per_state, actions_for_t
            )

        return treatment_effects

    @staticmethod
    def get_true_ate(action_idx, n_periods=120, obs_per_state=20, actions_for_t=np.inf):
        control_rewards = SepsisTrajectories.get_cumulative_reward_for_action(
            0, n_periods, obs_per_state, actions_for_t
        )
        treatment_rewards = SepsisTrajectories.get_cumulative_reward_for_action(
            action_idx, n_periods, obs_per_state, actions_for_t
        )

        return treatment_rewards - control_rewards

    @staticmethod
    def get_experimental_data(action_idx: int):
        states_t, actions_t, rewards_t = SepsisTrajectories.get_transitions(action_idx, 20)
        states_c, actions_c, rewards_c = SepsisTrajectories.get_transitions(0, 20)
        s0s = SepsisTrajectories.get_state_idxes_by_type(non_absorbing=True)
        s0s_vec = [MDP.State(state_idx, idx_type='full').get_full_state_vector() for state_idx in s0s]
        return (
            np.vstack(s0s_vec),
            np.vstack((states_c, states_t)),
            np.vstack((actions_c, actions_t)),
            np.vstack((rewards_c, rewards_t))
        )


if __name__ == "__main__":
    print(SepsisTrajectories.get_true_ates(120, actions_for_t=np.inf))
    # for action_idx in range(SepsisTrajectories.n_actions):
    #     SepsisTrajectories.get_experimental_data(action_idx)
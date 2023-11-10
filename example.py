from typing import Tuple, List

import numpy as np
import numpy.testing as npt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import td_model


class MarkovRewardProcess(object):
    eps = 1e-5

    def __init__(self, sigma, slope, gamma=0.9, treatment_effect_s=0.1, treatment_effect_r=0.1):
        self.sigma = sigma
        self.slope = slope
        self.gamma = gamma
        self.treatment_effect_states = treatment_effect_s
        self.treatment_effect_reward = treatment_effect_r
        self.reward_slope = 1
        self.reward_curvature = 0.5

    @staticmethod
    def featurize_sa(states, actions):
        ones = np.ones_like(actions)
        return np.hstack((states, actions, ones))

    def get_linear_wsa_from_rct_data(self, control_data, treatment_data, pi_a):
        states_control, rewards_control = control_data
        states_treatment, rewards_treatment = treatment_data
        n = states_control.shape[0] + states_treatment.shape[0]
        r = np.vstack((rewards_control[:, 0][:, None], rewards_treatment[:, 0][:, None]))
        s = np.vstack((states_control[:, 0][:, None], states_treatment[:, 0][:, None]))
        s_next = np.vstack((states_control[:, 1][:, None], states_treatment[:, 1][:, None]))
        actions = np.vstack(
            (np.zeros_like(states_control[:, 0][:, None]), np.ones_like(states_treatment[:, 0][:, None]))
        )
        a_under_policy = np.random.binomial(n=1, p=pi_a, size=actions.shape[0])[:, None]
        anext_under_policy = np.random.binomial(n=1, p=pi_a, size=actions.shape[0])[:, None]
        sa = MarkovRewardProcess.featurize_sa(s, actions)
        s_a0 = MarkovRewardProcess.featurize_sa(s, np.zeros_like(s))
        s_a1 = MarkovRewardProcess.featurize_sa(s, np.ones_like(s))
        s_a_pi_a = (1 - pi_a) * s_a0 + pi_a * s_a1
        s_next_a0 = MarkovRewardProcess.featurize_sa(s_next, np.zeros_like(s))
        s_next_a1 = MarkovRewardProcess.featurize_sa(s_next, np.ones_like(s))
        s_next_a_pi_a = (1 - pi_a) * s_next_a0 + pi_a * s_next_a1
        # s_a_pi_a = MarkovRewardProcess.featurize_sa(s, a_under_policy)
        # snext_a_pi_a = MarkovRewardProcess.featurize_sa(s_next, anext_under_policy)
        alpha_hat = np.linalg.solve(
            -self.gamma * s_next_a_pi_a.T.dot(sa) / n + sa.T.dot(sa) / n,
            (1 - self.gamma) * np.mean(s_a_pi_a, axis=0)
        )  # w(s, a) = (s, a, 1)' @ alpha_hat
        return alpha_hat

    def get_srs_next_from_rct_data(
            self, states_trajectory, rewards_trajectory, actions
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Only see the first transition
        states = states_trajectory[:, 0]
        states_next = states_trajectory[:, 1]
        r = rewards_trajectory[:, 0]
        return states, r, states_next, actions[:, 0]

    def generate_trajectory(
            self,
            treatment: bool,
            s0: np.ndarray,
            steps: int,
            max_t_for_treatment=np.inf,
            drift=0.
    ):

        n = len(s0)
        state_trajectory = np.zeros((n, steps + 1))
        rewards_trajectory = np.zeros((n, steps))

        state_trajectory[:, 0] = s0
        for t_idx in range(steps):
            # Simulate state transition based on Gaussian persistence model
            if treatment and t_idx <= max_t_for_treatment:
                next_states = state_trajectory[:, t_idx] + np.random.normal(self.treatment_effect_states, self.sigma, n)
            else:
                next_states = state_trajectory[:, t_idx] + np.random.normal(0, self.sigma, n)
            next_states = np.clip(next_states, 0, 1)
            state_trajectory[:, t_idx + 1] = next_states

            # Compute reward based on the linear reward function
            if treatment and t_idx <= max_t_for_treatment:
                rewards_trajectory[:, t_idx] = self.reward_slope * (state_trajectory[:, t_idx] ** self.reward_curvature) + drift * t_idx + self.treatment_effect_states
            else:
                rewards_trajectory[:, t_idx] = self.reward_slope * (state_trajectory[:, t_idx] ** self.reward_curvature) + drift * t_idx

        return state_trajectory, rewards_trajectory

    def rct_data(self, n=1000, steps=12, max_t_for_treatment=np.inf, drift=0.):
        s0 = np.random.uniform(0, 1, n)
        control_data = self.generate_trajectory(False, s0, steps, max_t_for_treatment, drift)
        treatment_data = self.generate_trajectory(True, s0, steps, max_t_for_treatment, drift)
        return s0, control_data, treatment_data

    @staticmethod
    def get_cumulative_reward(gamma, rewards: np.ndarray):
        T = rewards.shape[1]
        gammas = gamma ** np.arange(T)
        rewards_sum = np.multiply(gammas[None, :], rewards).sum(axis=1)
        normalization_factor = 1 - gamma
        return normalization_factor * rewards_sum

    def _get_ate(self, control_data, treatment_data):
        c_rewards = MarkovRewardProcess.get_cumulative_reward(self.gamma, control_data[1])
        t_rewards = MarkovRewardProcess.get_cumulative_reward(self.gamma, treatment_data[1])
        return t_rewards.mean() - c_rewards.mean()

    def fit_q(
        self,
        pi_a1: float,
        states_trajectory,
        rewards_trajectory,
        actions_trajectory,
        val_frac=0.2
    ):
        weights = np.ones_like(actions_trajectory[:, 0])
        weights[actions_trajectory[:, 0] == 0.] = 1 - pi_a1
        weights[actions_trajectory[:, 0] == 1.] = pi_a1

        pos_weights_mask = weights > 0
        states_trajectory = states_trajectory[pos_weights_mask]
        rewards_trajectory = rewards_trajectory[pos_weights_mask]
        actions_trajectory = actions_trajectory[pos_weights_mask]
        weights = weights[pos_weights_mask]

        s, r, s_next, a = self.get_srs_next_from_rct_data(
            states_trajectory, rewards_trajectory, actions_trajectory
        )
        if pi_a1 == 0.:
            assert a.mean() == 0.
        elif pi_a1 == 1.:
            assert a.mean() == 1.

        val_n = int(np.floor(s.shape[0] * val_frac))
        v_mlp = td_model.QModel(self.gamma)
        v_mlp.fit_model(
            pi_a1,
            s_next[val_n:, None], r[val_n:, None], s[val_n:, None], a[val_n:, None], weights[val_n:, None],
            s_next[:val_n, None], r[:val_n, None], s[:val_n, None], a[:val_n, None], weights[:val_n, None]
        )
        return v_mlp

    def doubly_robust_pot_outcome(self, s, a, w_alpha_hat, q_s0, q_error):
        w_sa = self.featurize_sa(s[:, None], a[:, None]).dot(w_alpha_hat)
        npt.assert_almost_equal(w_sa.mean(), 1., decimal=5)
        return (1 - self.gamma) * q_s0 + np.multiply(w_sa, q_error).mean()

    def get_ate_qw(self, s0, control_data, treatment_data, treatment_pi_a1):
        states = np.vstack((control_data[0], treatment_data[0]))
        rewards = np.vstack((control_data[1], treatment_data[1]))
        actions = np.vstack((np.zeros((control_data[0].shape[0], 1)), np.ones((treatment_data[0].shape[0], 1))))
        s, r, s_next, a = self.get_srs_next_from_rct_data(states, rewards, actions)

        p = np.random.permutation(len(states))

        mlp_c = self.fit_q(0, control_data[0], control_data[1], np.zeros_like(control_data[0]))
        control_q = mlp_c.predict(mlp_c.best_params, s0[:, None], np.zeros_like(s0[:, None]))
        control_td_error = mlp_c.get_td_error(
            mlp_c.best_params,
            self.gamma,
            control_data[0][:, 1][:, None],
            control_data[1][:, 0][:, None],
            control_data[0][:, 0][:, None],
            np.zeros((control_data[0].shape[0]))[:, None],
            0.
        )
        w_alpha_hat_0 = self.get_linear_wsa_from_rct_data(control_data, treatment_data, 0.)
        control_qw = self.doubly_robust_pot_outcome(s, a, w_alpha_hat_0, control_q.mean(), control_td_error)

        mlp_t = self.fit_q(treatment_pi_a1, states[p], rewards[p], actions[p])
        actions_under_pi = np.random.binomial(n=1, p=treatment_pi_a1, size=s0.shape[0])
        treatment_q = mlp_t.predict(mlp_t.best_params, s0[:, None], actions_under_pi)
        if treatment_pi_a1 == 1:
            treatment_td_error = mlp_t.get_td_error(
                mlp_t.best_params,
                self.gamma,
                treatment_data[0][:, 1][:, None],
                treatment_data[1][:, 0][:, None],
                treatment_data[0][:, 0][:, None],
                np.ones((treatment_data[0].shape[0]))[:, None],
                1.
            )
        else:
            treatment_td_error = mlp_t.get_td_error(
                mlp_t.best_params,
                self.gamma,
                s_next[:, None],
                r[:, None],
                s[:, None],
                a[:, None],
                treatment_pi_a1
            )
        w_alpha_hat_pi = self.get_linear_wsa_from_rct_data(control_data, treatment_data, treatment_pi_a1)
        treatment_qw = self.doubly_robust_pot_outcome(s, a, w_alpha_hat_pi, treatment_q.mean(), treatment_td_error)
        return treatment_qw - control_qw

    def fit_rf(self, control_data, features=None, rewards=None):
        rf = RandomForestRegressor()
        if rewards is not None:
            cumulative_reward = self.get_cumulative_reward(self.gamma, rewards)
        else:
            cumulative_reward = self.get_cumulative_reward(self.gamma, control_data[1][:, 1:])
        if features is not None:
            rf.fit(features, cumulative_reward)
        else:
            rf.fit(control_data[0][:, 1].reshape(-1, 1), cumulative_reward)
        return rf

    def get_naive_scaling_ate(self, control_data, treatment_data, T):
        first_T_periods_weights = (1 - self.gamma ** T) / (1 - self.gamma)
        treatment_r0 = treatment_data[1][:, 0].mean()
        control_r0 = control_data[1][:, 0].mean()
        return (1 - self.gamma) * first_T_periods_weights * (treatment_r0 - control_r0)

    def get_ate_rf(self, control_data, treatment_data):
        # surrogates method assume first treatment/surrogate is observed
        rf = self.fit_rf(control_data)
        treatment_r0 = treatment_data[1][:, 0] * (1 - self.gamma)
        control_r0 = control_data[1][:, 0] * (1 - self.gamma)
        treatment_rewards_hat = self.gamma * rf.predict(treatment_data[0][:, 1].reshape(-1, 1)) + treatment_r0
        control_rewards_hat = self.gamma * rf.predict(control_data[0][:, 1].reshape(-1, 1)) + control_r0
        return treatment_rewards_hat.mean() - control_rewards_hat.mean()

    def get_ate_rf_mid_surrogate(self, control_data, treatment_data, t_obs_surrogate):
        # surrogates method assume treatment/surrogate is observed upto (inclusive) t_obs_surrogate
        treatment_r0 = treatment_data[1][:, 0] * (1 - self.gamma)
        control_r0 = control_data[1][:, 0] * (1 - self.gamma)
        if t_obs_surrogate == 0:
            rf = self.fit_rf(control_data, features=control_data[0][:, 1].reshape(-1, 1))
            treatment_rewards_hat = self.gamma * rf.predict(treatment_data[0][:, 1].reshape(-1, 1)) + treatment_r0
            control_rewards_hat = self.gamma * rf.predict(control_data[0][:, 1].reshape(-1, 1)) + control_r0
        else:
            rf = self.fit_rf(
                control_data,
                features=control_data[0][:, t_obs_surrogate].reshape(-1, 1),
            )
            treatment_rewards_hat = self.gamma * rf.predict(treatment_data[0][:, t_obs_surrogate].reshape(-1, 1)) + treatment_r0
            control_rewards_hat = self.gamma * rf.predict(control_data[0][:, t_obs_surrogate].reshape(-1, 1)) + control_r0
        return treatment_rewards_hat.mean() - control_rewards_hat.mean()

    def get_ate_rf_mid_surrogate_and_reward(self, control_data, treatment_data, t_obs_surrogate):
        # surrogates method assume treatment/surrogate is observed upto (inclusive) t_obs_surrogate
        treatment_r0 = self.get_cumulative_reward(self.gamma, treatment_data[1][:, :t_obs_surrogate])
        control_r0 = self.get_cumulative_reward(self.gamma, control_data[1][:, :t_obs_surrogate])
        if t_obs_surrogate == 0:
            rf = self.fit_rf(
                control_data,
                features=control_data[0][:, 1].reshape(-1, 1),
                rewards=control_data[1][:, t_obs_surrogate:]
            )
            treatment_rewards_hat = self.gamma * rf.predict(treatment_data[0][:, 1].reshape(-1, 1)) + treatment_r0
            control_rewards_hat = self.gamma * rf.predict(control_data[0][:, 1].reshape(-1, 1)) + control_r0
        else:
            rf = self.fit_rf(
                control_data,
                features=control_data[0][:, t_obs_surrogate].reshape(-1, 1),
                rewards=control_data[1][:, t_obs_surrogate:]
            )
            treatment_rewards_hat = (self.gamma ** t_obs_surrogate) * rf.predict(treatment_data[0][:, t_obs_surrogate].reshape(-1, 1)) + treatment_r0
            control_rewards_hat = (self.gamma ** t_obs_surrogate) * rf.predict(control_data[0][:, t_obs_surrogate].reshape(-1, 1)) + control_r0
        return treatment_rewards_hat.mean() - control_rewards_hat.mean()

    @staticmethod
    def compare_under_short(n_treatment_periods=None):
        assert n_treatment_periods is not None
        if n_treatment_periods is not None:
            assert n_treatment_periods > 0
        m = MarkovRewardProcess(0.1, 1)
        n = 2000
        steps = 120

        s0, control_data, treatment_data = m.rct_data(
            n,
            steps,
            n_treatment_periods - 1 if n_treatment_periods < np.infty else np.infty,
            0
        )
        ate_true_drift = m._get_ate(control_data, treatment_data)
        ate_naive = m.get_naive_scaling_ate(control_data, treatment_data, n_treatment_periods)
        ate_rf = m.get_ate_rf(control_data, treatment_data)
        ate_mid_rf = m.get_ate_rf_mid_surrogate(
            control_data,
            treatment_data,
            int(np.floor(n_treatment_periods / 2)) if n_treatment_periods < np.infty else 120
        )
        ate_all_rf = m.get_ate_rf_mid_surrogate(
            control_data,
            treatment_data,
            n_treatment_periods if n_treatment_periods < np.infty else 120
        )
        ate_mid_both_rf = m.get_ate_rf_mid_surrogate_and_reward(
            control_data,
            treatment_data,
            int(np.floor(n_treatment_periods / 2)) if n_treatment_periods < np.infty else 120
        )
        ate_all_both_rf = m.get_ate_rf_mid_surrogate_and_reward(
            control_data,
            treatment_data,
            n_treatment_periods if n_treatment_periods < np.infty else 120
        )
        ate_qw = m.get_ate_qw(
            s0,
            control_data,
            treatment_data,
            1 - m.gamma ** n_treatment_periods if n_treatment_periods < np.infty else 1.
        )
        print(
            f'{n_treatment_periods}-period ATE: {ate_true_drift:.3f}, '
            f'All-Surrogate: {ate_all_rf:.3f}, {ate_all_both_rf:.3f}, '
            f'Mid-Surrogate: {ate_mid_rf:.3f}, {ate_mid_both_rf:.3f}, '
            f'Estimate: {ate_rf:.3f}, '
            f'Naive: {ate_naive:.3f}, '
            f'Q Estimate: {ate_qw:.3f}'
        )
        return {
            'T': n_treatment_periods,
            'ate': ate_true_drift,
            'ate_naive': ate_naive,
            'ate_surrogate': ate_rf,
            'ate_mid_surrogate': ate_mid_rf,
            'ate_all_surrogate': ate_all_rf,
            'ate_mid_both_surrogate': ate_mid_both_rf,
            'ate_all_both_surrogate': ate_all_both_rf,
            'ate_qw': ate_qw
        }

    @staticmethod
    def monte_carlo(n_reps: int, Ts: List[float]):
        results = []
        for _ in range(n_reps):
            for T in Ts:
                results.append(MarkovRewardProcess.compare_under_short(T))

        pd.DataFrame(results).to_csv('qw_vs_surrogates.csv', index=False)


if __name__ == "__main__":
    # MarkovRewardProcess.compare_under_permanent()
    np.random.seed(2023)
    Ts = [1, 6, 12, 24, 48, np.infty][::-1]
    MarkovRewardProcess.monte_carlo(n_reps=20, Ts=Ts)

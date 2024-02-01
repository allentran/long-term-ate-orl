from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import td_model
from sepsis_sim import trajectories


class Estimators(object):
    def __init__(self, gamma):
        self.gamma = gamma

    @staticmethod
    def featurize_sa(states, actions, polynomial=True):
        ones = np.ones_like(actions)
        if polynomial:
            return np.hstack(
                (
                    states, states ** 2, states ** 3,
                    actions * states, actions * states ** 2, actions * states ** 3,
                    actions,
                    ones
                )
            )
        else:
            return np.hstack((states, actions, actions * states, ones))

    def get_linear_wsa_from_rct_data(self, s, s_next, a, pi_a, polynomial=True):
        n = s.shape[0]
        sa = Estimators.featurize_sa(s, a[:, 0][:, None], polynomial)
        s_a0 = Estimators.featurize_sa(s, np.zeros_like(s)[:, 0][:, None], polynomial)
        s_a1 = Estimators.featurize_sa(s, np.ones_like(s)[:, 0][:, None], polynomial)
        s_a_pi_a = (1 - pi_a) * s_a0 + pi_a * s_a1
        s_next_a0 = Estimators.featurize_sa(s_next, np.zeros_like(s)[:, 0][:, None], polynomial)
        s_next_a1 = Estimators.featurize_sa(s_next, np.ones_like(s)[:, 0][:, None], polynomial)
        s_next_a_pi_a = (1 - pi_a) * s_next_a0 + pi_a * s_next_a1
        alpha_hat = np.linalg.solve(
            -self.gamma * s_next_a_pi_a.T.dot(sa) / n + sa.T.dot(sa) / n,
            (1 - self.gamma) * np.mean(s_a_pi_a, axis=0)
        )  # w(s, a) = (s, a, 1)' @ alpha_hat
        return alpha_hat

    def fit_rf(self, control_data, features=None, rewards=None):
        rf = RandomForestRegressor()
        if rewards is not None:
            cumulative_reward = MarkovDecisionProcess.get_cumulative_reward(self.gamma, rewards)
        else:
            cumulative_reward = MarkovDecisionProcess.get_cumulative_reward(self.gamma, control_data[1][:, 1:])
        if features is not None:
            rf.fit(features, cumulative_reward)
        else:
            rf.fit(control_data[0][:, 1].reshape(-1, 1), cumulative_reward)
        return rf

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
        treatment_r0 = MarkovDecisionProcess.get_cumulative_reward(self.gamma, treatment_data[1][:, :t_obs_surrogate])
        control_r0 = MarkovDecisionProcess.get_cumulative_reward(self.gamma, control_data[1][:, :t_obs_surrogate])
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

    def get_ate_from_trajectory(self, control_data, treatment_data, return_all=False):
        c_rewards = MarkovDecisionProcess.get_cumulative_reward(self.gamma, control_data[1])
        t_rewards = MarkovDecisionProcess.get_cumulative_reward(self.gamma, treatment_data[1])
        if return_all:
            return t_rewards.mean(), c_rewards.mean()
        return t_rewards.mean() - c_rewards.mean()

    def fit_q(
        self,
        pi_a1: float,
        s,
        r,
        s_next,
        a,
        is_nonabs,
        is_nonabs_next,
        val_frac=0.2,
        batch_size=None,
        learning_rate=1e-2,
        init_params=None
    ):

        v_mlp = td_model.QModel(self.gamma)
        v_mlp.fit_model(
            pi_a1,
            s_next, r[:, None], s, a[:, None], is_nonabs[:, None], is_nonabs_next[:, None],
            batch_size=batch_size, val_frac=val_frac, learning_rate=learning_rate, init_params=init_params
        )
        return v_mlp

    def doubly_robust_pot_outcome(self, s, a, alpha_hat, q_s0, q_error, polynomial=True):
        w_sa = self.featurize_sa(s, a[:, 0][:, None], polynomial).dot(alpha_hat)
        assert q_error.shape[0] == w_sa.shape[0]
        return (1 - self.gamma) * q_s0 + np.multiply(w_sa, q_error).mean()

    def get_ate_qw(
        self,
        s0, s, r, s_next, a, is_nonabs, is_nonabs_next, treatment_pi_a1,
        k_folds=5,
        return_all=False,
        batch_size=None,
        val_frac=None,
        polynomial=True,
        learning_rate=1e-2
    ):
        kfold_idxes = np.random.choice(k_folds, size=s.shape[0])
        control_qws = []
        treatment_qws = []
        for k_fold_idx in range(k_folds):
            train = kfold_idxes != k_fold_idx
            mlp_c = self.fit_q(
                0,
                s[train], r[train], s_next[train], a[train], is_nonabs[train], is_nonabs_next[train],
                val_frac=val_frac, batch_size=batch_size, learning_rate=learning_rate
            )
            control_q = mlp_c.predict(mlp_c.best_params, s0, np.zeros_like(s0[:, None]), np.ones_like(s0[:, None]))
            control_td_error = mlp_c.get_td_error(
                mlp_c.best_params,
                self.gamma,
                s_next[~train],
                r[~train, None],
                s[~train],
                a[~train, None],
                is_nonabs[~train, None],
                is_nonabs_next[~train, None],
                0.
            )
            alpha_hat_0 = self.get_linear_wsa_from_rct_data(
                s[train], s_next[train], a[train], 0., polynomial
            )
            control_qw = self.doubly_robust_pot_outcome(
                s[~train], a[~train], alpha_hat_0, control_q.mean(), control_td_error, polynomial
            )
            control_qws.append(float(control_qw))

            mlp_t = self.fit_q(
                treatment_pi_a1,
                s[train], r[train], s_next[train], a[train], is_nonabs[train], is_nonabs_next[train],
                val_frac=val_frac, batch_size=batch_size, learning_rate=learning_rate,
                init_params=mlp_c.best_params
            )
            actions_under_pi = np.random.binomial(n=1, p=treatment_pi_a1, size=s0.shape[0])
            treatment_q = mlp_t.predict(mlp_t.best_params, s0, actions_under_pi, np.ones_like(s0[:, None]))
            treatment_td_error = mlp_t.get_td_error(
                mlp_t.best_params,
                self.gamma,
                s_next[~train],
                r[~train, None],
                s[~train],
                a[~train, None],
                is_nonabs[~train, None],
                is_nonabs_next[~train, None],
                treatment_pi_a1
            )
            alpha_hat_pi = self.get_linear_wsa_from_rct_data(
                s[train], s_next[train], a[train], treatment_pi_a1, polynomial
            )
            treatment_qw = self.doubly_robust_pot_outcome(
                s[~train], a[~train], alpha_hat_pi, treatment_q.mean(), treatment_td_error, polynomial
            )
            treatment_qws.append(treatment_qw)
        if return_all:
            return np.mean(treatment_qws), np.mean(control_qws)
        return np.mean(treatment_qws) - np.mean(control_qws)

    def get_ate_qw_from_trajectories(
        self, s0, control_data, treatment_data, treatment_pi_a1, return_all=False, polynomial=True
    ):
        T = control_data[1].shape[1]
        states = np.vstack((control_data[0], treatment_data[0]))
        rewards = np.vstack((control_data[1], treatment_data[1]))
        actions = np.vstack((np.zeros((control_data[0].shape[0], T)), np.ones((treatment_data[0].shape[0], T))))
        s, r, s_next, a = MarkovDecisionProcess.get_srs_next_from_rct_data(states, rewards, actions)
        return self.get_ate_qw(
            s0[:, None], s[:, None], r, s_next[:, None], a, treatment_pi_a1, return_all=return_all, polynomial=polynomial
        )

    def get_naive_scaling_ate(self, control_data, treatment_data, T):
        first_T_periods_weights = (1 - self.gamma ** T) / (1 - self.gamma)
        treatment_r0 = treatment_data[1][:, 0].mean()
        control_r0 = control_data[1][:, 0].mean()
        return (1 - self.gamma) * first_T_periods_weights * (treatment_r0 - control_r0)


class MarkovDecisionProcess(object):
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
    def get_srs_next_from_rct_data(states_trajectory, rewards_trajectory, actions) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
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
        hidden_state=False,
        rho=0.
    ):
        n = len(s0)
        n_states = 2 if hidden_state else 1
        state_trajectory = np.zeros((n, n_states, steps + 1))
        rewards_trajectory = np.zeros((n, steps))
        state_trajectory[:, :, 0] = s0
        for t_idx in range(steps):
            # Simulate state transition based on Gaussian persistence model
            add_treatment_effect = treatment and t_idx <= max_t_for_treatment
            noise = np.random.normal(0, self.sigma, (n, n_states))
            curr_state = state_trajectory[:, :, t_idx]
            if hidden_state:
                cov = np.array([[1 - rho, 0.], [rho, 1.]])
                curr_state = curr_state.dot(cov)
            if add_treatment_effect:
                next_states = curr_state + noise + self.treatment_effect_states
            else:
                next_states = curr_state + noise
            next_states = np.clip(next_states, 0, 1)
            state_trajectory[:, :, t_idx + 1] = next_states

            # Compute reward based on the linear reward function
            if treatment and t_idx <= max_t_for_treatment:
                rewards_trajectory[:, t_idx] = self.reward_slope * (state_trajectory[:, 0, t_idx] ** self.reward_curvature) + self.treatment_effect_states
            else:
                rewards_trajectory[:, t_idx] = self.reward_slope * (state_trajectory[:, 0, t_idx] ** self.reward_curvature)

        if hidden_state:
            return state_trajectory[:, 0, :], rewards_trajectory
        return np.squeeze(state_trajectory), rewards_trajectory

    def rct_data(self, n=1000, steps=12, max_t_for_treatment=np.inf):
        s0 = np.random.uniform(0, 1, n)[:, None]
        control_data = self.generate_trajectory(False, s0, steps, max_t_for_treatment)
        treatment_data = self.generate_trajectory(True, s0, steps, max_t_for_treatment)
        return s0, control_data, treatment_data

    def rct_data_2d(self, n=1000, steps=12, max_t_for_treatment=np.inf, corr=0.):
        s0 = np.random.uniform(0, 1, (n, 2))
        control_data = self.generate_trajectory(False, s0, steps, max_t_for_treatment, True, corr)
        treatment_data = self.generate_trajectory(True, s0, steps, max_t_for_treatment, True, corr)
        return s0[:, 0][:, None], control_data, treatment_data

    @staticmethod
    def get_cumulative_reward(gamma, rewards: np.ndarray):
        T = rewards.shape[1]
        gammas = gamma ** np.arange(T)
        rewards_sum = np.multiply(gammas[None, :], rewards).sum(axis=1)
        normalization_factor = 1 - gamma
        return normalization_factor * rewards_sum


class Experiments(object):

    def __init__(self, gamma, n_units=1000):
        self.estimators = Estimators(gamma)
        self.n_units = n_units

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

    def hidden_state_xp(self, n_treatment_periods, corr: float):
        assert n_treatment_periods is not None
        assert n_treatment_periods > 0
        m = MarkovDecisionProcess(0.1, 1)
        steps = 120

        s0, control_data, treatment_data = m.rct_data_2d(
            self.n_units,
            steps,
            n_treatment_periods - 1 if n_treatment_periods < np.infty else np.infty,
            corr
        )
        t_mean, c_mean = self.estimators.get_ate_from_trajectory(control_data, treatment_data, return_all=True)
        t_qw_mean, c_qw_mean = self.estimators.get_ate_qw_from_trajectories(
            s0,
            control_data,
            treatment_data,
            1 - m.gamma ** n_treatment_periods if n_treatment_periods < np.infty else 1.,
            return_all=True,
            polynomial=True
        )
        return {
            'T': n_treatment_periods,
            'treatment_mean': t_mean,
            'control_mean': c_mean,
            'corr': corr,
            'treatment_qw_mean': t_qw_mean,
            'control_qw_mean': c_qw_mean,
        }

    def coverage_xp(self, n_treatment_periods, filter_prob: float):
        assert n_treatment_periods is not None
        assert n_treatment_periods > 0
        m = MarkovDecisionProcess(0.1, 1)
        steps = 120

        s0, control_data, treatment_data = m.rct_data(
            self.n_units,
            steps,
            n_treatment_periods - 1 if n_treatment_periods < np.infty else np.infty,
        )
        ate_true_drift = self.estimators.get_ate_from_trajectory(control_data, treatment_data)
        control_data = Experiments.filter_by_initial_state(control_data, filter_prob, 0.2, 0.8)
        treatment_data = Experiments.filter_by_initial_state(treatment_data, filter_prob,  0.2, 0.8)
        ate_qw = self.estimators.get_ate_qw_from_trajectories(
            s0,
            control_data,
            treatment_data,
            1 - m.gamma ** n_treatment_periods if n_treatment_periods < np.infty else 1.,
            polynomial=True
        )
        print(
            f'{n_treatment_periods}-period ATE: {ate_true_drift:.3f}, '
            f'Q Estimate: {ate_qw:.3f}'
        )
        return {
            'T': n_treatment_periods,
            'ate': ate_true_drift,
            'ate_qw': ate_qw,
            'filter_prob': filter_prob
        }

    def get_true_ate_sepsis(self, action_idx, n_treatment_periods):
        return trajectories.SepsisTrajectories.get_true_ate(
            action_idx, obs_per_state=40, actions_for_t=n_treatment_periods
        )

    def compare_sepsis_estimators(
            self, action_idx, n_treatment_periods=None, batch_size=2000, val_frac=0.0, learning_rate=1e-2
    ):
        pi_action = 1 - self.estimators.gamma ** n_treatment_periods if n_treatment_periods < np.infty else 1.
        s0, states, actions, rewards, is_nonabs = trajectories.SepsisTrajectories.get_experimental_data(
            action_idx, hidden=False
        )
        treatment_effect = self.estimators.get_ate_qw(
            s0,
            states[:, :, 0],
            rewards,
            states[:, :, 1],
            actions,
            is_nonabs[:, 0],
            is_nonabs[:, 1],
            pi_action,
            val_frac=val_frac,
            batch_size=batch_size,
            polynomial=False,
            learning_rate=learning_rate
        )
        return {
            'T': n_treatment_periods,
            'ate_qw': treatment_effect,
            'action': action_idx
        }

    def compare_estimators(self, n_treatment_periods=None):
        assert n_treatment_periods is not None
        if n_treatment_periods is not None:
            assert n_treatment_periods > 0
        m = MarkovDecisionProcess(0.1, 1)
        steps = 120

        s0, control_data, treatment_data = m.rct_data(
            self.n_units,
            steps,
            n_treatment_periods - 1 if n_treatment_periods < np.infty else np.infty,
        )
        ate_true_drift = self.estimators.get_ate_from_trajectory(control_data, treatment_data)
        ate_naive = self.estimators.get_naive_scaling_ate(control_data, treatment_data, n_treatment_periods)
        ate_rf = self.estimators.get_ate_rf(control_data, treatment_data)
        ate_mid_rf = self.estimators.get_ate_rf_mid_surrogate(
            control_data,
            treatment_data,
            int(np.floor(n_treatment_periods / 2)) if n_treatment_periods < np.infty else 120
        )
        ate_all_rf = self.estimators.get_ate_rf_mid_surrogate(
            control_data,
            treatment_data,
            n_treatment_periods if n_treatment_periods < np.infty else 120
        )
        ate_mid_both_rf = self.estimators.get_ate_rf_mid_surrogate_and_reward(
            control_data,
            treatment_data,
            int(np.floor(n_treatment_periods / 2)) if n_treatment_periods < np.infty else 120
        )
        ate_all_both_rf = self.estimators.get_ate_rf_mid_surrogate_and_reward(
            control_data,
            treatment_data,
            n_treatment_periods if n_treatment_periods < np.infty else 120
        )
        ate_qw = self.estimators.get_ate_qw_from_trajectories(
            s0,
            control_data,
            treatment_data,
            1 - m.gamma ** n_treatment_periods if n_treatment_periods < np.infty else 1.,
            polynomial=True
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


class MonteCarlo(object):

    def __init__(self, gamma):
        self.experiments = Experiments(gamma)

    def hidden_state_experiment_monte_carlo(self, n_reps):
        results = []
        T = np.inf
        for corr in [0, 0.1, 0.2, 0.5, 1]:
            for _ in range(n_reps):
                results.append(self.experiments.hidden_state_xp(T, corr))

        pd.DataFrame(results).to_csv('hidden-state.xp.csv', index=False)

    def coverage_experiment_monte_carlo(self, n_reps):
        results = []
        T = 12
        for prob_scale in [0.01, 0.1, 0.2, 0.5, 1]:
            for _ in range(n_reps):
                results.append(self.experiments.coverage_xp(T, prob_scale))

        pd.DataFrame(results).to_csv('coverage.xp.csv', index=False)

    def vary_treatment_horizon_experiment(self, n_reps: int, Ts: List[float]):
        results = []
        for T in Ts:
            for _ in range(n_reps):
                results.append(self.experiments.compare_estimators(T))

        pd.DataFrame(results).to_csv('qw_vs_surrogates.csv', index=False)


if __name__ == "__main__":
    np.random.seed(2023)
    # Experiments(0.9).use_of_T_data(np.inf, 3)
    gamma = 0.9
    monte_carlo = MonteCarlo(gamma)
    Ts = [1, 6, 12, 24, 48, np.infty][::-1]
    monte_carlo.vary_treatment_horizon_experiment(n_reps=20, Ts=Ts)
    # monte_carlo.coverage_experiment_monte_carlo(20)
    # monte_carlo.hidden_state_experiment_monte_carlo(20)

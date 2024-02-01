import itertools

import numpy as np
import pandas as pd
from metaflow import FlowSpec, step, titus, Parameter, resources, conda, conda_base, retry
from simulation import Experiments


@conda_base(
    libraries={
        "jax": "0.4.20",
        "scikit-learn": "1.3.2",
        "numpy": "1.25.2",
        "optax": "0.1.7",
        "flax": "0.7.5",
        "matplotlib": "3.8.1",
        "pandas": "2.1.2",
    }
)
class SepsisLongTermTreatmentsFlow(FlowSpec):

    gamma = Parameter('gamma', type=float, default=0.9)
    n_runs = Parameter('n_runs', type=int, default=100)
    batch_size = Parameter('batch_size', type=int)
    val_frac = Parameter('val_frac', type=float)
    learning_rate = Parameter('learning_rate', type=float)

    @conda(disabled=True)
    @step
    def start(self):
        self.T = np.inf
        self.experiments = Experiments(self.gamma)
        self.runs = list(range(self.n_runs))
        self.next(
            # self.coverage_xp_monte_carlo,
            # self.hidden_state_xp_monte_carlo,
            self.treatment_duration_xp_monte_carlo
            # self.trajectory_xp_monte_carlo
        )

    # @conda(disabled=True)
    # @step
    # def coverage_xp_monte_carlo(self):
    #     self.prob_scale = [0.01, 0.1, 0.2, 0.5, 1] * self.n_runs
    #     self.next(self.coverage_xp, foreach='prob_scale')
    #
    # @titus
    # @retry
    # @resources(cpu=16)
    # @step
    # def coverage_xp(self):
    #     self.results = self.monte_carlo.experiments.coverage_xp(self.T, self.input)
    #     self.next(self.merge_coverage_xp)
    #
    # @conda(disabled=True)
    # @step
    # def merge_coverage_xp(self, coverage_xps_at_params_inputs):
    #     results = []
    #     for coverage_xps_at_params_input in coverage_xps_at_params_inputs:
    #         results.append(coverage_xps_at_params_input.results)
    #     self.coverage_xp_results = pd.DataFrame(results).assign(experiment='coverage')
    #     self.next(self.join)

    # @conda(disabled=True)
    # @step
    # def hidden_state_xp_monte_carlo(self):
    #     self.corr = [0, 0.1, 0.2, 0.5, 1] * self.n_runs
    #     self.next(self.hidden_state_xp, foreach='corr')
    #
    # @titus
    # @retry
    # @resources(cpu=16)
    # @step
    # def hidden_state_xp(self):
    #     self.results = self.monte_carlo.experiments.hidden_state_xp(self.T, self.input)
    #     self.next(self.merge_hidden_state_xp)
    #
    # @conda(disabled=True)
    # @step
    # def merge_hidden_state_xp(self, hidden_state_xps_at_params_inputs):
    #     results = []
    #     for hidden_state_xp_input in hidden_state_xps_at_params_inputs:
    #         results.append(hidden_state_xp_input.results)
    #     self.hidden_state_results = pd.DataFrame(results).assign(experiment='hidden_state_no_bias')
        # self.next(self.join)
        # self.next(self.end)

    @conda(disabled=True)
    @step
    def treatment_duration_xp_monte_carlo(self):
        durations_and_actions = list(
            itertools.product([1, 6, 12, 24, 48, np.infty], list(range(1, 8)))
        )
        self.durations_and_actions_x_runs = durations_and_actions * self.n_runs
        self.true_ates = {}
        for duration, action in durations_and_actions:
            self.true_ates[(duration, action)] = self.experiments.get_true_ate_sepsis(action, duration)
        self.next(self.treatment_duration_xp, foreach='durations_and_actions_x_runs')

    @titus
    @retry
    @resources(cpu=4)
    @step
    def treatment_duration_xp(self):
        duration, action_idx = self.input
        self.results = self.experiments.compare_sepsis_estimators(
            action_idx, duration, batch_size=self.batch_size, val_frac=self.val_frac, learning_rate=self.learning_rate
        )
        self.results['ate'] = self.true_ates[(duration, action_idx)]
        self.next(self.merge_duration_xp)

    @conda(disabled=True)
    @step
    def merge_duration_xp(self, duration_xps_inputs):
        results = []
        for duration_xp_input in duration_xps_inputs:
            results.append(duration_xp_input.results)
        self.duration_results = pd.DataFrame(results).assign(experiment='duration')
        self.next(self.end)

    # @conda(disabled=True)
    # @step
    # def trajectory_xp_monte_carlo(self):
    #     self.Ts = [3, 6, 12, 24, 48] * self.n_runs
    #     self.next(self.trajectory_xp, foreach='Ts')
    #
    # @titus
    # @resources(cpu=16)
    # @step
    # def trajectory_xp(self):
    #     self.results = self.monte_carlo.experiments.use_of_T_data(np.inf, self.input)
    #     self.next(self.merge_trajectory_xp)
    #
    # @conda(disabled=True)
    # @step
    # def merge_trajectory_xp(self, trajectory_xps_inputs):
    #     results = []
    #     for trajectory_xp_input in trajectory_xps_inputs:
    #         results.append(trajectory_xp_input.results)
    #     self.trajectory_results = pd.DataFrame(results).assign(experiment='trajectory')
    #     self.next(self.join)

    # @conda(disabled=True)
    # @step
    # def join(self, inputs):
    #     self.merge_artifacts(inputs)
    #     self.next(self.end)

    @conda(disabled=True)
    @step
    def end(self):
        pass


if __name__ == "__main__":
    SepsisLongTermTreatmentsFlow()


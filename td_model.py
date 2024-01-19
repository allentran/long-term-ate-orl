from typing import Sequence
from functools import partial

import optax
import numpy as np
from flax import linen as nn
import jax
from jax import numpy as jnp


class DataLoader:
    def __init__(self, states, next_states, rewards, actions, validation_frac=0.2):

        # Shuffle data
        indices = np.arange(len(states))
        np.random.shuffle(indices)

        # Split data into training and validation sets
        split_idx = int(len(states) * (1 - validation_frac))
        train_indices, val_indices = indices[:split_idx], indices[split_idx:]

        # Training data
        self.train_states = states[train_indices]
        self.train_next_states = next_states[train_indices]
        self.train_rewards = rewards[train_indices]
        self.train_actions = actions[train_indices]

        if validation_frac > 0:
            # Validation data
            self.val_states = states[val_indices]
            self.val_next_states = next_states[val_indices]
            self.val_rewards = rewards[val_indices]
            self.val_actions = actions[val_indices]

    def get_random_batch(self, batch_size=None):
        if batch_size:
            indices = np.random.choice(len(self.train_states), batch_size, replace=False)
            batch_states = self.train_states[indices]
            batch_next_states = self.train_next_states[indices]
            batch_rewards = self.train_rewards[indices]
            batch_actions = self.train_actions[indices]
            return batch_states, batch_next_states, batch_rewards, batch_actions
        else:
            return self.train_states, self.train_next_states, self.train_rewards, self.train_actions

    def get_validation_batch(self):
        return self.val_states, self.val_next_states, self.val_rewards, self.val_actions


class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat, use_bias=False) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            if i == len(self.layers) - 1:  # skip connection ensures a linear last layer of the og input
                x = jnp.concatenate([inputs, x], axis=-1)
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.sigmoid(x)
        return x


class QModel(object):
    def __init__(self, gamma, features=(128, 64, 1)):
        assert features[-1] == 1

        self.features = features
        self.gamma = gamma
        self.min_loss = np.inf

    def fit_model(
            self,
            pi_a1: float,
            s_next: np.ndarray, r: np.ndarray, s: np.ndarray, a: np.ndarray,
            batch_size=None,
            val_frac=None
    ):
        mlp = ExplicitMLP(self.features)
        optimizer = optax.adam(1e-2)
        l2_penalty = 0#1e-3

        @jax.jit
        def _td_pred(params, s):
            return mlp.apply(params, s)

        @partial(jax.jit)
        def _td_pred_given_action(params, s, a):
            return jnp.where(
                a == 0,
                _td_pred(params[0], s),
                _td_pred(params[1], s)
            )

        @jax.jit
        def vmap_td_pred(params, s, a):
            # td_pred_t = jax.vmap(_td_pred, in_axes=(None, 0))(params[1], s[:, None])
            # return jnp.where(a == 0., td_pred_c, td_pred_t)
            return jax.vmap(_td_pred_given_action, in_axes=(None, 0, 0))(params, s, a)

        @partial(jax.jit, static_argnums=2)
        def _td_pred_target(params, target_params, gamma, s_tplus1, r_t, s_t, a_t, pi_a1):
            q_sa_t = _td_pred_given_action(params, s_t, a_t)
            q_s_tplus1_a0 = _td_pred(target_params[0], s_tplus1)
            q_s_tplus1_a1 = _td_pred(target_params[1], s_tplus1)
            target = r_t + gamma * ((1 - pi_a1) * q_s_tplus1_a0 + pi_a1 * q_s_tplus1_a1)

            return q_sa_t, target

        @partial(jax.jit)
        def vmap_td_error(params, gamma, snext, r, s, a, pi_a1):
            vmap_pred_target = jax.vmap(_td_pred_target, in_axes=(None, None, None, 0, 0, 0, 0, None))
            q_sa, target = vmap_pred_target(params, params, gamma, snext, r, s, a, pi_a1)
            return target - q_sa

        @partial(jax.jit, static_argnums=2)
        def _td_loss(params, target_params, gamma, s_tplus1, r_t, s_t, a_t, pi_a1):
            q_t, target = _td_pred_target(params, target_params, gamma, s_tplus1, r_t, s_t, a_t, pi_a1)
            squared_loss = jnp.squeeze(((jax.lax.stop_gradient(target) - q_t) ** 2))
            return squared_loss

        @partial(jax.jit, static_argnums=2)
        def vmap_td_loss(params, target_params, gamma, s_tplus1, r_t, s_t, a_t, pi_a1):
            def l2_loss(x):
                return l2_penalty * (x ** 2).mean()
            f = jax.vmap(_td_loss, in_axes=(None, None, None, 0, 0, 0, 0, None))
            l2_loss_0 = sum(l2_loss(w) for w in jax.tree_util.tree_leaves(params[0]["params"]))
            l2_loss_1 = sum(l2_loss(w) for w in jax.tree_util.tree_leaves(params[1]["params"]))
            per_obs_loss = f(params, target_params, gamma, s_tplus1, r_t, s_t, a_t, pi_a1)
            return jnp.mean(per_obs_loss) + l2_loss_0 + l2_loss_1

        @jax.jit
        def step(params, target_params, opt_state, gamma, s_next, r, s, a, pi_a1):
            loss_value, grads = jax.value_and_grad(vmap_td_loss, argnums=0)(
                params, target_params, gamma, s_next, r, s, a, pi_a1
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, opt_state, loss_value

        self.predict = vmap_td_pred
        self.get_td_error = vmap_td_error
        key1, key2 = jax.random.split(jax.random.PRNGKey(0), 2)
        treatment_params = mlp.init(key1, s[0])
        control_params = mlp.init(key2, s[0])
        mlp_params = (control_params, treatment_params)
        best_params = mlp_params
        opt_state = optimizer.init(mlp_params)

        iters_since_best_loss = 0
        cnt = 0
        best_params_updated = False
        update_target_steps = 16
        target_params = mlp_params

        data_loader = DataLoader(s, s_next, r, a, validation_frac=val_frac)
        if val_frac > 0:
            val_s, val_s_next, val_r, val_a = data_loader.get_validation_batch()

        while True:
            cnt += 1
            # w = 0.9
            # target_params = jax.tree_map(lambda x, y: w * x + (1. - w) * y, target_params, mlp_params)
            if cnt % update_target_steps == 0:
                target_params = mlp_params
            iters_since_best_loss += 1
            # Perform one gradient update.
            batch_s, batch_s_next, batch_r, batch_a = data_loader.get_random_batch(batch_size)
            mlp_params, opt_state, loss = step(
                mlp_params,
                target_params,
                opt_state,
                self.gamma,
                batch_s_next[:, None],
                batch_r[:, None],
                batch_s[:, None],
                batch_a[:, None],
                pi_a1
            )
            if cnt % 66 == 0:
                if val_frac > 0:
                    val_loss = vmap_td_loss(
                        mlp_params, target_params, self.gamma, val_s_next[:, None], val_r[:, None], val_s[:, None], val_a[:, None], pi_a1
                    )
                else:
                    val_loss = loss
                if val_loss < self.min_loss and cnt > 1000:
                    self.min_loss = val_loss
                    best_params = mlp_params
                    iters_since_best_loss = 0
                    best_params_updated = True

                # print(
                #     f'\rLoss step {cnt}: {loss:.5f} - {val_loss:.5f} Best={self.min_loss:.5f}',
                #     end='',
                #     flush=True
                # )
            if iters_since_best_loss > 10000 or self.min_loss < 1e-05 or cnt > 6000:
                # print(self.min_loss)
                break

        assert best_params_updated
        self.target_params = target_params
        self.best_params = best_params

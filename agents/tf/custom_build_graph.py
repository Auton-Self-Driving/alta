from stable_baselines.deepq.build_graph import *

# imports from build_graph
import tensorflow as tf
from gym.spaces import MultiDiscrete

from stable_baselines.common import tf_util



def custom_build_train(q_func, ob_space, ac_space, optimizer, sess, grad_norm_clipping=None,
                gamma=1.0, double_q=True, scope="deepq", reuse=None,
                param_noise=False, param_noise_filter_func=None, full_tensorboard_log=False, n_step=1):
    """
    Creates the train function:

    :param q_func: (DQNPolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param reuse: (bool) whether or not to reuse the graph variables
    :param optimizer: (tf.train.Optimizer) optimizer to use for the Q-learning objective.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param grad_norm_clipping: (float) clip gradient norms to this value. If None no clipping is performed.
    :param gamma: (float) discount rate.
    :param double_q: (bool) if true will use Double Q Learning (https://arxiv.org/abs/1509.06461). In general it is a
        good idea to keep it enabled.
    :param scope: (str or VariableScope) optional scope for variable_scope.
    :param reuse: (bool) whether or not the variables should be reused. To be able to reuse the scope must be given.
    :param param_noise: (bool) whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    :param param_noise_filter_func: (function (TensorFlow Tensor): bool) function that decides whether or not a
        variable should be perturbed. Only applicable if param_noise is True. If set to None, default_param_noise_filter
        is used by default.
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param n_step: value of n in n-step return version of DQN. n=1 corresponds to standard DQN.

    :return: (tuple)

        act: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor) function to select and action given
            observation. See the top of the file for details.
        train: (function (Any, numpy float, numpy float, Any, numpy bool, numpy float): numpy float)
            optimize the error in Bellman's equation. See the top of the file for details.
        update_target: (function) copy the parameters from optimized Q function to the target Q function.
            See the top of the file for details.
        step_model: (DQNPolicy) Policy for evaluation
    """
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    with tf.variable_scope("input", reuse=reuse):
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

    with tf.variable_scope(scope, reuse=reuse):
        if param_noise:
            act_f, obs_phs = build_act_with_param_noise(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess,
                                                        param_noise_filter_func=param_noise_filter_func)
        else:
            act_f, obs_phs = build_act(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess)

        # q network evaluation
        with tf.variable_scope("step_model", reuse=True, custom_getter=tf_util.outer_scope_getter("step_model")):
            step_model = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True, obs_phs=obs_phs)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")
        # target q network evaluation

        with tf.variable_scope("target_q_func", reuse=False):
            target_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=tf.get_variable_scope().name + "/target_q_func")

        # compute estimate of best possible value starting from state at t + 1
        double_q_values = None
        double_obs_ph = target_policy.obs_ph
        if double_q:
            with tf.variable_scope("double_q", reuse=True, custom_getter=tf_util.outer_scope_getter("double_q")):
                double_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True)
                double_q_values = double_policy.q_values
                double_obs_ph = double_policy.obs_ph

    with tf.variable_scope("loss", reuse=reuse):
        # set up placeholders
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(step_model.q_values * tf.one_hot(act_t_ph, n_actions), axis=1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_best_using_online_net = tf.argmax(double_q_values, axis=1)
            q_tp1_best = tf.reduce_sum(target_policy.q_values * tf.one_hot(q_tp1_best_using_online_net, n_actions), axis=1)
        else:
            q_tp1_best = tf.reduce_max(target_policy.q_values, axis=1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # CUSTOM change: (gamma ** n_step)
        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + (gamma ** n_step) * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = tf_util.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        tf.summary.scalar("td_error", tf.reduce_mean(td_error))
        tf.summary.scalar("loss", weighted_error)

        if full_tensorboard_log:
            tf.summary.histogram("td_error", td_error)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # compute optimization op (potentially with gradient clipping)
        gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
        if grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)

                    if full_tensorboard_log:
                        tf.summary.histogram(var.name + '/gradient', gradients[i])

        params = tf.trainable_variables()
        if full_tensorboard_log:
            for var in params:
                tf.summary.histogram(var.name, var)

    with tf.variable_scope("input_info", reuse=False):
        tf.summary.scalar('rewards', tf.reduce_mean(rew_t_ph))
        tf.summary.scalar('importance_weights', tf.reduce_mean(importance_weights_ph))

        if full_tensorboard_log:
            tf.summary.histogram('rewards', rew_t_ph)
            tf.summary.histogram('importance_weights', importance_weights_ph)
            # if tf_util.is_image(obs_phs[0]):
            #     tf.summary.image('observation', obs_phs[0])
            # elif len(obs_phs[0].shape) == 1:
            #     tf.summary.histogram('observation', obs_phs[0])

    optimize_expr = optimizer.apply_gradients(gradients)

    summary = tf.summary.merge_all()

    # Create callable functions
    train = tf_util.function(
        inputs=[
            obs_phs[0],
            act_t_ph,
            rew_t_ph,
            target_policy.obs_ph,
            double_obs_ph,
            done_mask_ph,
            importance_weights_ph
        ],
        outputs=[summary, td_error],
        updates=[optimize_expr]
    )
    update_target = tf_util.function([], [], updates=[update_target_expr])

    return act_f, train, update_target, step_model


def custom_build_train_clipped_DDQN(q_func, ob_space, ac_space, optimizer, optimizer2, sess, grad_norm_clipping=None,
                gamma=1.0, double_q=True, scope="deepq", reuse=None,
                param_noise=False, param_noise_filter_func=None, full_tensorboard_log=False, n_step=1):
    """
    Creates the train function:

    :param q_func: (DQNPolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param reuse: (bool) whether or not to reuse the graph variables
    :param optimizer: (tf.train.Optimizer) optimizer to use for the Q-learning objective.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param grad_norm_clipping: (float) clip gradient norms to this value. If None no clipping is performed.
    :param gamma: (float) discount rate.
    :param double_q: (bool) if true will use Double Q Learning (https://arxiv.org/abs/1509.06461). In general it is a
        good idea to keep it enabled.
    :param scope: (str or VariableScope) optional scope for variable_scope.
    :param reuse: (bool) whether or not the variables should be reused. To be able to reuse the scope must be given.
    :param param_noise: (bool) whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    :param param_noise_filter_func: (function (TensorFlow Tensor): bool) function that decides whether or not a
        variable should be perturbed. Only applicable if param_noise is True. If set to None, default_param_noise_filter
        is used by default.
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param n_step: value of n in n-step return version of DQN. n=1 corresponds to standard DQN.

    :return: (tuple)

        act: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor) function to select and action given
            observation. See the top of the file for details.
        train: (function (Any, numpy float, numpy float, Any, numpy bool, numpy float): numpy float)
            optimize the error in Bellman's equation. See the top of the file for details.
        update_target: (function) copy the parameters from optimized Q function to the target Q function.
            See the top of the file for details.
        step_model: (DQNPolicy) Policy for evaluation
    """
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    with tf.variable_scope("input", reuse=reuse):
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

    with tf.variable_scope(scope, reuse=reuse):
        if param_noise:
            act_f, obs_phs = build_act_with_param_noise(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess,
                                                        param_noise_filter_func=param_noise_filter_func)
        else:
            act_f, obs_phs = build_act(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess)

        # q network evaluation
        with tf.variable_scope("step_model", reuse=True, custom_getter=tf_util.outer_scope_getter("step_model")):
            step_model = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True, obs_phs=obs_phs)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")
        # target q network evaluation

        with tf.variable_scope("target_q_func", reuse=False):
            target_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=tf.get_variable_scope().name + "/target_q_func")

        # compute estimate of best possible value starting from state at t + 1
        double_q_values = None
        double_obs_ph = target_policy.obs_ph
        if double_q:
            with tf.variable_scope("double_q", reuse=True, custom_getter=tf_util.outer_scope_getter("double_q")):
                double_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True)
                double_q_values = double_policy.q_values
                double_obs_ph = double_policy.obs_ph
        
        # CLIPPED CHANGE:
        # Declare step_model2 and target_policy2
        # TODO: Check scope to get tf.get_collection
        with tf.variable_scope("q2"):
            # q network evaluation
            with tf.variable_scope("step_model2", reuse=False, custom_getter=tf_util.outer_scope_getter("step_model2")):
                step_model2 = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False)
            q_func_vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")
            
            # target q network evaluation
            with tf.variable_scope("target_q_func2", reuse=False):
                target_policy2 = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False)
            target_q_func_vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope=tf.get_variable_scope().name + "/target_q_func2")

            # compute estimate of best possible value starting from state at t + 1
            double_q_values2 = None
            double_obs_ph2 = target_policy2.obs_ph
            if double_q:
                with tf.variable_scope("double_q2", reuse=True, custom_getter=tf_util.outer_scope_getter("double_q2")):
                    double_policy2 = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True)
                    double_q_values2 = double_policy2.q_values
                    double_obs_ph2 = double_policy2.obs_ph

        
        # # q network evaluation
        # with tf.variable_scope("step_model2", reuse=False):
        #     step_model2 = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False)
        # q_func_vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/step_model2/model")
        
        # # target q network evaluation
        # with tf.variable_scope("target_q_func2", reuse=False):
        #     target_policy2 = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False)
        # target_q_func_vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        #                                     scope=tf.get_variable_scope().name + "/target_q_func2")

        # # compute estimate of best possible value starting from state at t + 1
        # double_q_values2 = None
        # double_obs_ph2 = target_policy2.obs_ph
        # if double_q:
        #     with tf.variable_scope("random", reuse=True):
        #         double_policy2 = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True)
        #         double_q_values2 = double_policy2.q_values
        #         double_obs_ph2 = double_policy2.obs_ph


    with tf.variable_scope("loss", reuse=reuse):
        # set up placeholders
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(step_model.q_values * tf.one_hot(act_t_ph, n_actions), axis=1)

        # q scores for actions which we know were selected in the given state.
        q_t_selected2 = tf.reduce_sum(step_model2.q_values * tf.one_hot(act_t_ph, n_actions), axis=1)

        # target1 = r(s,a) + (1-done) * gamma * min (i=1, i=2) { Qi_target (s',a1')}
        # target2 = r(s,a) + (1-done) * gamma * min (i=1, i=2) { Qi_target (s',a2')}

        # t1 = min (i=1, i=2) { Qi_target (s',a1')}
        # t2 = min (i=1, i=2) { Qi_target (s',a2')}

        # where

        # ai' = argmax (a') (Qi_current (s'))

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            
            #a1'
            q_tp1_best_using_online_net = tf.argmax(double_q_values, axis=1)

            #a2'
            q_tp1_best_using_online_net2 = tf.argmax(double_q_values2, axis=1)


            q_tp1_best_Q1_a1 = tf.reduce_sum(target_policy.q_values * tf.one_hot(q_tp1_best_using_online_net, n_actions), axis=1)
            q_tp1_best_Q2_a1 = tf.reduce_sum(target_policy2.q_values * tf.one_hot(q_tp1_best_using_online_net, n_actions), axis=1)

            q_tp1_best_Q1_a2 = tf.reduce_sum(target_policy.q_values * tf.one_hot(q_tp1_best_using_online_net2, n_actions), axis=1)
            q_tp1_best_Q2_a2 = tf.reduce_sum(target_policy2.q_values * tf.one_hot(q_tp1_best_using_online_net2, n_actions), axis=1)

            t1 = tf.minimum(q_tp1_best_Q1_a1, q_tp1_best_Q2_a1)
            t2 = tf.minimum(q_tp1_best_Q1_a2, q_tp1_best_Q2_a2)

            
            # q_tp1_best = tf.reduce_sum(target_policy.q_values * tf.one_hot(q_tp1_best_using_online_net, n_actions), axis=1)
        else:
            q_tp1_best = tf.reduce_max(target_policy.q_values, axis=1)

        # q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best



        # CUSTOM change: (gamma ** n_step)
        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + (gamma ** n_step) * (1.0 - done_mask_ph) * t1

        q_t_selected_target2 = rew_t_ph + (gamma ** n_step) * (1.0 - done_mask_ph) * t2

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = tf_util.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        td_error2 = q_t_selected2 - tf.stop_gradient(q_t_selected_target2)
        errors2 = tf_util.huber_loss(td_error2)
        weighted_error2 = tf.reduce_mean(importance_weights_ph * errors2)

        tf.summary.scalar("td_error", tf.reduce_mean(td_error))
        tf.summary.scalar("loss", weighted_error)

        tf.summary.scalar("td_error2", tf.reduce_mean(td_error2))
        tf.summary.scalar("loss2", weighted_error2)

        if full_tensorboard_log:
            tf.summary.histogram("td_error", td_error)
            tf.summary.histogram("td_error2", td_error2)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # For Q2
        update_target_expr2 = []
        for var, var_target in zip(sorted(q_func_vars2, key=lambda v: v.name),
                                   sorted(target_q_func_vars2, key=lambda v: v.name)):
            update_target_expr2.append(var_target.assign(var))
        update_target_expr2 = tf.group(*update_target_expr2)

        # compute optimization op (potentially with gradient clipping)
        gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
        if grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)

                    if full_tensorboard_log:
                        tf.summary.histogram(var.name + '/gradient', gradients[i])
                        print(var.name)

        # Q2 : compute optimization op (potentially with gradient clipping)
        gradients2 = optimizer2.compute_gradients(weighted_error2, var_list=q_func_vars2)
        if grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients2):
                if grad is not None:
                    gradients2[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)

                    if full_tensorboard_log:
                        tf.summary.histogram(var.name + '/gradient2', gradients[i])
                        print(var.name)


        params = tf.trainable_variables()
        if full_tensorboard_log:
            for var in params:
                tf.summary.histogram(var.name, var)

    with tf.variable_scope("input_info", reuse=False):
        tf.summary.scalar('rewards', tf.reduce_mean(rew_t_ph))
        tf.summary.scalar('importance_weights', tf.reduce_mean(importance_weights_ph))

        if full_tensorboard_log:
            tf.summary.histogram('rewards', rew_t_ph)
            tf.summary.histogram('importance_weights', importance_weights_ph)
            # if tf_util.is_image(obs_phs[0]):
            #     tf.summary.image('observation', obs_phs[0])
            # elif len(obs_phs[0].shape) == 1:
            #     tf.summary.histogram('observation', obs_phs[0])

    optimize_expr = optimizer.apply_gradients(gradients)
    optimize_expr2 = optimizer2.apply_gradients(gradients2)

    summary = tf.summary.merge_all()

    # Create callable functions
    train = tf_util.function(
        inputs=[
            obs_phs[0],
            step_model2.obs_ph,
            act_t_ph,
            rew_t_ph,
            target_policy.obs_ph,
            double_obs_ph,
            target_policy2.obs_ph,
            double_obs_ph2,
            done_mask_ph,
            importance_weights_ph
        ],
        outputs=[summary, td_error],
        updates=[optimize_expr, optimize_expr2]
    )
    update_target = tf_util.function([], [], updates=[update_target_expr, update_target_expr2])

    return act_f, train, update_target, step_model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import ipdb
trace = ipdb.set_trace

import sys, os, glob

sys.path.append('/zfsauton2/home/vkadi/visdom/py/')
sys.path.append('./../../')
sys.path.append(os.path.abspath(os.path.join('../../', 'config')))
import visdom

from environment.carla_9_4.env import CarlaEnv
from environment.carla_9_4.config import ConfigManager
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
import tensorflow as tf
import gym

import matplotlib.pyplot as plt
import plotly.tools as tls

from plot_rewards_dynamic_actors import plot_runs

EPS = 1e-6
scale = 1e-2

def gaussian_likelihood(input_, mu_, log_std):
    pre_sum = -0.5 * (((input_ - mu_) / (np.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return np.sum(pre_sum, axis=1)

def apply_squashing_func(mu_, pi_, logp_pi):
    deterministic_policy = np.tanh(mu_)
    policy = np.tanh(pi_)
    upd_logp_pi = logp_pi-np.sum(np.log(1 - policy ** 2 + EPS), axis=1)
    return deterministic_policy, policy, upd_logp_pi

def vis_states(vis, states, plot_3d = False, save_path = ''):
	if plot_3d:
		if save_path!='' and os.path.exists(save_path):
			out = np.load(save_path)
		else:
			out = vis.embeddings(states, np.ones(len(states)), ret_3d = True, \
							opts = {'title': 'State space', 'width':1200, 'height':600})

		if save_path!='':
			np.save(save_path, np.asarray(out))
		vis.scatter(out, opts = {'title': 'State space','width':1200, 'height':600})
	else:
		vis.embeddings(states, np.ones(len(states)), opts = {'title': 'State space','width':1200, 'height':600})

def vis_actions(vis, actions):
	#vis.scatter(actions, opts = {'title': 'Action space', 'width':1200, 'height':600})
	vis.line(X = np.arange(len(actions[:,1])), Y = actions[:,1], opts = {'title': 'Speed_values', 'width':1200, 'height':600})

def vis_rewards(vis, rewards, mode = 'hist', st = 0, ed= -1):
	if mode=='hist':
		vis.histogram(X=rewards, opts = {'numbins':20, 'title': 'Histogram of Rewards', 'width':1200, 'height':600})
	else:
		vis.line(X = np.arange(len(rewards[st:ed])), Y = rewards[st:ed], \
							opts = {'title': 'Histogram of Rewards', 'width':1200, 'height':600})

def vis_V_values(vis, values, st = 0, ed= -1):
	vis.line(X = np.arange(len(values[st:ed])), Y = values[st:ed], opts = {'title': 'V_values', 'width':1200, 'height':600})

def vis_dones(vis, dones, st = 0, ed= -1):
	vis.line(X = np.arange(len(dones[st:ed])), Y = dones[st:ed], opts = {'title': 'Dones', 'width':1200, 'height':600})

def vis_policyloss(vis, model_dir_path, replay_buffer, fname=None, tseries = False, gpu='3', st = 0, ed= -1):
	states = []
	actions = []
	rewards = []
	next_states = []
	dones = []

	for elt in replay_buffer:
		states.append(elt[0])
		actions.append(elt[1])
		rewards.append(elt[2])
		next_states.append(elt[3])
		dones.append(elt[4])

	states = np.array(states)
	actions = np.array(actions)
	rewards = np.array(rewards)
	next_states = np.array(next_states)
	dones = np.array(dones)


	config = ConfigManager(algo="SAC")
	config.config["carla_gpu"] = gpu
	config.config["code_gpu"]  = gpu
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(config.config["code_gpu"])
	config.config["input_type"] = "wp_obs_info_speed_steer_ldist_goal_light"
	config.config["task"] = "self-driving"
	config.config["network"] = "2_layer"
	config.config["ent_coef"] = -1
	config.config["n_steps"] = 1
	config.config["train_freq"] = 1
	config.config["gradient_steps_per_iteration"] = 1
	config.config["target_update_interval"] = 1

	from my_sac import MY_SAC

	ALTA_LOGS = model_dir_path

	set_global_seeds(5)

	env = CarlaEnv(config=config.config, vis_wrapper=None, logger=None, log_dir = ALTA_LOGS, base_prefix = None, prefix = None)
	dummy_env = DummyVecEnv([lambda: env])

	#MODEL_PATH = ALTA_LOGS+'/sac_weights3000000.pkl'
	filenames = []
	if tseries:
		for fl in sorted(os.listdir(model_dir_path)):
			if fl[-4:]=='.pkl':
				filenames.append(fl)
	else:
		filenames.append(fname)

	qloss_hist = []
	for filename in filenames[-10:]:
		MODEL_PATH = os.path.join(ALTA_LOGS, filename)

		model = MY_SAC.load(MODEL_PATH, env)

		'''for samp in replay_buffer:
			self.replay_buffer.add(samp[0], samp[1], samp[2], samp[3], samp[4])'''

		batch_sz = 2048
		n_batches = int(len(replay_buffer)/batch_sz)

		with tf.variable_scope("loss", reuse=True):
			q_backup = tf.stop_gradient(model.rewards_ph + (1 - model.terminals_ph) * model.gamma * model.value_target)

		q1_values = np.zeros((len(replay_buffer), ))
		stp1_values = np.zeros((len(replay_buffer), ))
		q1_loss = np.zeros((len(replay_buffer), ))
		q2_loss = np.zeros((len(replay_buffer), ))
		policy_loss = []
		terminal_state_loss = 0.0
		terminal_states = 0
		nonterminal_state_loss = 0.0
		nonterminal_states = 0
		for idx in range(n_batches+1):
			if(idx%1000==0):
				print(idx,'/', n_batches)
			batch_obs = states[idx*batch_sz:min((idx+1)*batch_sz, len(replay_buffer)), :]
			batch_actions = actions[idx*batch_sz:min((idx+1)*batch_sz, len(replay_buffer)), :]
			batch_rewards = rewards[idx*batch_sz:min((idx+1)*batch_sz, len(replay_buffer))]
			batch_next_obs = next_states[idx*batch_sz:min((idx+1)*batch_sz, len(replay_buffer)), :]
			batch_dones = dones[idx*batch_sz:min((idx+1)*batch_sz, len(replay_buffer))]

			batch_pred_actions = model.predict(batch_obs, deterministic=True)[0]

			feed_dict = {
				model.observations_ph: batch_obs,
				#model.actions_ph: batch_pred_actions,
				model.actions_ph: batch_actions,
				model.next_observations_ph: batch_next_obs,
				model.rewards_ph: batch_rewards.reshape(len(batch_rewards), -1),
				model.terminals_ph: batch_dones.reshape(len(batch_dones), -1),
				model.learning_rate_ph: 0.0
			}

			target_q = model.sess.run(q_backup, feed_dict)
			next_s_val = model.sess.run(model.value_target, feed_dict)
			out = model.sess.run(model.step_ops, feed_dict)
			batch_policy_loss, qf1_loss, qf2_loss, value_loss, qf1, qf2, *values = out
			qf1_loss2 = (target_q-qf1)**2
			qf2_loss2 = (target_q-qf2)**2

			done_idxs = np.where(batch_dones==1)[0]
			terminal_state_loss += np.sum(qf1_loss2[done_idxs]) 
			terminal_states += len(done_idxs)
			nondone_idxs = np.where(batch_dones<1)[0]
			nonterminal_state_loss += np.sum(qf1_loss2[nondone_idxs]) 
			nonterminal_states += len(nondone_idxs)			

			q1_loss[idx*batch_sz:min((idx+1)*batch_sz, len(replay_buffer))] = np.squeeze(qf1_loss2)
			q2_loss[idx*batch_sz:min((idx+1)*batch_sz, len(replay_buffer))] = np.squeeze(qf2_loss2)
			q1_values[idx*batch_sz:min((idx+1)*batch_sz, len(replay_buffer))] = np.squeeze(qf1)
			stp1_values[idx*batch_sz:min((idx+1)*batch_sz, len(replay_buffer))] = np.squeeze(next_s_val)
			policy_loss.append(batch_policy_loss)
			#out = model._train_step(step, None, 0.0)
		#qloss_hist.append(np.mean(q1_loss))
		policy_loss = np.asarray(policy_loss)
		qloss_hist.append(np.mean(policy_loss))
		#trace()

	if tseries:
		vis.line(X = np.arange(len(qloss_hist)), Y = qloss_hist, opts = {'title': 'policy_loss', 'width':1200, 'height':600})
		print(qloss_hist)
	else:
		#s = 0
		#e = -1
		#vis.line(X = np.arange(len(q1_loss[st:ed])), Y = q1_loss[st:ed], opts = {'title': 'q1Loss', 'width':1200, 'height':600})
		vis.line(X = np.arange(len(q1_values[st:ed])), Y = q1_values[st:ed], opts = {'title': 'q1Values', 'width':1200, 'height':600})
		#vis.line(X = np.arange(len(stp1_values[st:ed])), Y = stp1_values[st:ed], opts = {'title': 'V_Values', 'width':1200, 'height':600})
		print(ALTA_LOGS[-5:])
		print((nonterminal_state_loss*1.0)/nonterminal_states, nonterminal_states)
		print((terminal_state_loss*1.0)/terminal_states, terminal_states)
		#vis.line(X = np.arange(len(q2_loss)), Y = q2_loss, opts = {'title': 'q2Loss', 'width':1200, 'height':600})


	env.close()

def compute_done_idxs(states):
	delta_s = np.abs(states[1:, :]- states[:-1, :])
	delta_s = np.sum(delta_s, axis = -1)
	done_idxs = []
	for i in range(len(delta_s)-1):
		if(delta_s[i+1]>0 and delta_s[i]==0):
			done_idxs.append(i)
	return done_idxs

def vis_qvalues2(vis, model_dir_path, replay_buffer, fname=None, tseries = False, gpu='3', st = 0, ed= -1):
	run_ids = np.arange(5)+1
	states = []
	actions = []
	rewards = []
	next_states = []
	dones = []

	for elt in replay_buffer:
		states.append(elt[0])
		actions.append(elt[1])
		rewards.append(elt[2])
		next_states.append(elt[3])
		dones.append(elt[4])

	states = np.array(states)
	actions = np.array(actions)
	rewards = np.array(rewards)
	next_states = np.array(next_states)
	dones = np.array(dones)
	done_idxs = np.where(dones==1)[0]
	#trace()
	#done_idxs = compute_done_idxs(states)

	config = ConfigManager(algo="SAC")
	config.config["carla_gpu"] = gpu
	config.config["code_gpu"]  = gpu
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(config.config["code_gpu"])
	config.config["input_type"] = "wp_obs_info_speed_steer_ldist_goal_light"
	config.config["task"] = "self-driving"
	config.config["network"] = "2_layer"
	config.config["ent_coef"] = -1
	config.config["n_steps"] = 1
	config.config["train_freq"] = 1
	config.config["gradient_steps_per_iteration"] = 1
	config.config["target_update_interval"] = 1

	from my_sac import MY_SAC

	#ALTA_LOGS = os.path.basename(model_dir_path)
	ALTA_LOGS = model_dir_path[:-1].split('/')[-1]

	set_global_seeds(5)

	env = CarlaEnv(config=config.config, vis_wrapper=None, logger=None, log_dir = os.path.join(model_dir_path, ALTA_LOGS+'_runid_run'+str(1)), base_prefix = None, prefix = None)
	dummy_env = DummyVecEnv([lambda: env])

	#MODEL_PATH = ALTA_LOGS+'/sac_weights3000000.pkl'
	filenames = []
	if tseries:
		filename_prefix = 'sac_weights'
		stepsize = 75000
		ctr = 1
		unsorted_files = os.listdir(os.path.join(model_dir_path, ALTA_LOGS+'_runid_run'+str(1)))
		unsorted_files = [fl for fl in unsorted_files if (fl[-4:]=='.pkl' and fl[:len(filename_prefix)]==filename_prefix)]
		num_files = len(unsorted_files)
		while len(filenames)<num_files:
			curr_filename = filename_prefix+str(ctr*stepsize)+'.pkl'
			if curr_filename in unsorted_files:
				filenames.append(curr_filename)
			ctr+=1
	else:
		filenames.append(fname)

	qloss_hist = []
	for filename in filenames:
		file_terminal_qloss = []
		for run_id in run_ids:
			MODEL_PATH = os.path.join(os.path.join(model_dir_path, ALTA_LOGS+'_runid_run'+str(run_id)), filename)
			model = MY_SAC.load(MODEL_PATH, env)

			'''for samp in replay_buffer:
				self.replay_buffer.add(samp[0], samp[1], samp[2], samp[3], samp[4])'''


			batch_sz = 1024
			n_batches = int(len(replay_buffer)/batch_sz)

			with tf.variable_scope("loss", reuse=True):
				q_backup = tf.stop_gradient(model.rewards_ph + (1 - model.terminals_ph) * model.gamma * model.value_target)

			q1_values = np.zeros((len(replay_buffer), ))
			stp1_values = np.zeros((len(replay_buffer), ))
			q1_loss = np.zeros((len(replay_buffer), ))
			q2_loss = np.zeros((len(replay_buffer), ))

			terminal_qloss = []

			terminal_state_loss = 0.0
			terminal_states = 0
			nonterminal_state_loss = 0.0
			nonterminal_states = 0
			#trace()
			penultimate_state_value = 0.0
			for i in range(len(done_idxs))[:-1]:
				batch_obs = states[done_idxs[i]+1:done_idxs[i+1]+1, :]
				batch_actions = actions[done_idxs[i]+1:done_idxs[i+1]+1, :]
				batch_rewards = rewards[done_idxs[i]+1:done_idxs[i+1]+1]
				batch_next_obs = next_states[done_idxs[i]+1:done_idxs[i+1]+1, :]
				batch_dones = dones[done_idxs[i]+1:done_idxs[i+1]+1]

				batch_pred_actions = model.predict(batch_obs, deterministic=True)[0]

				feed_dict = {
					model.observations_ph: batch_obs,
					#model.actions_ph: batch_pred_actions,
					model.actions_ph: batch_actions,
					model.next_observations_ph: batch_next_obs,
					model.rewards_ph: batch_rewards.reshape(len(batch_rewards), -1),
					model.terminals_ph: batch_dones.reshape(len(batch_dones), -1),
					model.learning_rate_ph: 0.0
				}

				target_q = model.sess.run(q_backup, feed_dict)
				next_s_val = model.sess.run(model.value_target, feed_dict)
				out = model.sess.run(model.step_ops, feed_dict)
				policy_loss, qf1_loss, qf2_loss, value_loss, qf1, qf2, *values = out
				qf1_loss2 = (target_q-qf1)**2
				qf2_loss2 = (target_q-qf2)**2

				'''batch_done_idxs = np.where(batch_dones==1)[0]
				terminal_state_loss += np.sum(qf1_loss2[batch_done_idxs]) 
				terminal_states += len(batch_done_idxs)
				nondone_idxs = np.where(batch_dones<1)[0]
				nonterminal_state_loss += np.sum(qf1_loss2[nondone_idxs]) 
				nonterminal_states += len(nondone_idxs)'''

				q1_loss[done_idxs[i]+1:done_idxs[i+1]+1] = np.squeeze(qf1_loss2)
				q2_loss[done_idxs[i]+1:done_idxs[i+1]+1] = np.squeeze(qf2_loss2)
				q1_values[done_idxs[i]+1:done_idxs[i+1]+1] = np.squeeze(qf1)
				stp1_values[done_idxs[i]+1:done_idxs[i+1]+1] = np.squeeze(next_s_val)
				#out = model._train_step(step, None, 0.0)

				terminal_qloss.append(qf1_loss2[-1])

				if(len(qf1_loss2)<2):
					continue

				'''vis.line(X = np.arange(len(qf1_loss2)), Y = np.squeeze(qf1_loss2), opts = {'title': 'q1loss', 'width':1200, 'height':600})
				vis.line(X = np.arange(len(qf1)), Y = np.squeeze(qf1), opts = {'title': 'q1Values', 'width':1200, 'height':600})
				vis.line(X = np.arange(len(next_s_val)), Y =  np.squeeze(next_s_val),\
											opts = {'title': 'V_Values', 'width':1200, 'height':600})
				vis.line(X = np.arange(len(batch_dones)), Y = np.squeeze(batch_dones),\
											opts = {'title': 'dones', 'width':1200, 'height':600})
				#vis.line(X = np.arange(len(policy_loss)), Y = np.squeeze(policy_loss), opts = {'title': 'KL_loss', 'width':1200, 'height':600})
				vis.line(X = np.arange(len(batch_rewards)), Y = np.squeeze(batch_rewards),\
											opts = {'title': 'rewards', 'width':1200, 'height':600})
				trace()'''

			terminal_qloss = np.asarray(terminal_qloss)
			file_terminal_qloss.append(np.mean(terminal_qloss))

		file_terminal_qloss = np.asarray(file_terminal_qloss)
		qloss_hist.append(np.mean(file_terminal_qloss))
			#trace()

	if tseries:
		vis.line(X = np.arange(len(qloss_hist)), Y = qloss_hist, opts = {'title': 'q1Loss', 'width':1200, 'height':600})
		print(qloss_hist)
	else:
		#s = 0
		#e = -1
		#vis.line(X = np.arange(len(q1_loss[st:ed])), Y = q1_loss[st:ed], opts = {'title': 'q1Loss', 'width':1200, 'height':600})
		vis.line(X = np.arange(len(q1_values[st:ed])), Y = q1_values[st:ed], opts = {'title': 'q1Values', 'width':1200, 'height':600})
		#vis.line(X = np.arange(len(stp1_values[st:ed])), Y = stp1_values[st:ed], opts = {'title': 'V_Values', 'width':1200, 'height':600})
		print(ALTA_LOGS[-5:])
		print((nonterminal_state_loss*1.0)/nonterminal_states, nonterminal_states)
		print((terminal_state_loss*1.0)/terminal_states, terminal_states)
		#vis.line(X = np.arange(len(q2_loss)), Y = q2_loss, opts = {'title': 'q2Loss', 'width':1200, 'height':600})


	env.close()


def vis_exp_qvalues(args, vis, model_dir_path, replay_buffer, fname=None, tseries = False, gpu='3', st = 0, ed= -1):
	states = []
	actions = []
	rewards = []
	next_states = []
	dones = []

	for elt in replay_buffer:
		states.append(elt[0])
		actions.append(elt[1])
		rewards.append(elt[2])
		next_states.append(elt[3])
		dones.append(elt[4])

	states = np.array(states)
	actions = np.array(actions)
	rewards = np.array(rewards)
	next_states = np.array(next_states)
	dones = np.array(dones)
	done_idxs = np.where(dones==1)[0]

	config = ConfigManager(algo="SAC")
	config.config["carla_gpu"] = gpu
	config.config["code_gpu"]  = gpu
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(config.config["code_gpu"])
	config.config["input_type"] = "wp_obs_info_speed_steer_ldist_goal_light"
	config.config["task"] = "self-driving"
	config.config["network"] = "2_layer"
	config.config["ent_coef"] = -1
	config.config["n_steps"] = 1
	config.config["train_freq"] = 1
	config.config["gradient_steps_per_iteration"] = 1
	config.config["target_update_interval"] = 1

	from my_sac import MY_SAC

	ALTA_LOGS = model_dir_path

	set_global_seeds(5)

	env = CarlaEnv(config=config.config, vis_wrapper=None, logger=None, log_dir = ALTA_LOGS, base_prefix = None, prefix = None)
	dummy_env = DummyVecEnv([lambda: env])

	#MODEL_PATH = ALTA_LOGS+'/sac_weights3000000.pkl'
	filenames = []
	if tseries:
		for fl in sorted(os.listdir(model_dir_path)):
			if fl[-4:]=='.pkl':
				filenames.append(fl)
	else:
		filenames.append(fname)

	expected_qval_hist = []
	for filename in filenames:
		MODEL_PATH = os.path.join(ALTA_LOGS, filename)

		model = MY_SAC.load(MODEL_PATH, env)

		traj_probs = np.zeros((len(done_idxs)-1, 1))

		traj_rewards = np.zeros((len(done_idxs)-1, ))

		metric_vals = np.zeros((len(done_idxs)-1, ))

		for i in range(len(done_idxs))[:-1]:
			batch_obs = states[done_idxs[i]+1:done_idxs[i+1]+1, :]
			#trace()
			flag = (actions[done_idxs[i]+1:done_idxs[i+1]+1, :]>0)*1.0
			flag = flag*EPS*-1.0
			flag[flag==0] += EPS
			batch_actions = np.arctanh(actions[done_idxs[i]+1:done_idxs[i+1]+1, :]+flag)
			batch_rewards = rewards[done_idxs[i]+1:done_idxs[i+1]+1]
			batch_next_obs = next_states[done_idxs[i]+1:done_idxs[i+1]+1, :]
			batch_dones = dones[done_idxs[i]+1:done_idxs[i+1]+1]

			batch_pred_params = model.policy_tf.proba_step(batch_obs)
			batch_pred_prob = gaussian_likelihood(batch_actions, batch_pred_params[0], np.log(batch_pred_params[1]))
			_, _, logp_pi = apply_squashing_func(batch_pred_params[0], batch_actions, batch_pred_prob)

			#metric = np.exp(np.sum(logp_pi))*np.sum(batch_rewards)
			traj_rewards[i] = np.sum(batch_rewards)
			traj_probs[i] = np.sum(logp_pi)/(1e+5+1)

		max_val = np.max(traj_probs)
		'''denom_exp_q1_vals = traj_probs - traj_probs.T
		denom_exp_q1_vals = np.exp(denom_exp_q1_vals)
		denom_exp_q1_vals = np.sum(denom_exp_q1_vals, axis = -1)
		metric = np.mean(denom_exp_q1_vals*traj_rewards)'''
		mod_traj_probs = traj_probs-max_val
		softmax_traj_probs = np.exp(mod_traj_probs)/np.sum(np.exp(mod_traj_probs))
		metric = np.mean(softmax_traj_probs*traj_rewards)
		#trace()
		

		expected_qval_hist.append(np.mean(metric))

	vis.line(X = np.arange(len(expected_qval_hist)), Y = expected_qval_hist, opts = {'title': 'exp_qvals_'+os.path.basename(args.path), 'width':1200, 'height':600})
	print(expected_qval_hist)

	env.close()

def vis_validation(vis, path):
	mpl_fig = plt.figure()

	exp_folder = '/zfsauton2/home/vkadi/projects/alta/alta-logs/test'

	exp = os.path.dirname(path)

	plot_runs(0, exp_folder, exp, 'black', 200, None, 'sac')

	axes = plt.gca()
	#axes.set_ylim(bottom=0, top = 120000)
	#plt.legend(loc='lower right', prop={'size' : 36})

	vis.matplot(plt, opts = {'title': 'Val plots', 'width':1200, 'height':600})

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Visualize buffer')
	parser.add_argument('--path', default='', type=str, metavar='PATH', help='path to the buffer file (default: none)')
	parser.add_argument('--exp-path', default='', type=str, metavar='PATH', help='path to the model folder (default: none)')
	parser.add_argument('--model-name', default='', type=str, metavar='PATH', help='name of saved policy file (default: none)')
	parser.add_argument('--tag', default='', type=str, metavar='TAG', help='Custom tag')
	parser.add_argument('-s', '--states', dest='states', action='store_true', help='visualize states')
	parser.add_argument('-a', '--actions', dest='actions', action='store_true', help='visualize actions')
	parser.add_argument('-r', '--rewards', dest='rewards', action='store_true', help='visualize rewards')
	parser.add_argument('-q', '--qvalues', dest='qvalues', action='store_true', help='visualize qvalues')
	parser.add_argument('-t', '--tseries', dest='tseries', action='store_true', help='plot tderror')
	parser.add_argument('-n', '--next-states', dest='next_states', action='store_true', help='visualize next states')
	parser.add_argument('-e', '--dones', dest='dones', action='store_true', help='visualize dones')
	parser.add_argument('-d', '--dim', dest='dim', default=2, type=int, help='num of downsampled state dimensions')
	parser.add_argument('-v', '--val', dest='val', action='store_true', help='visualize validation plots')
	parser.add_argument('-g', '--gpu', dest='gpu', default='3', type=str, help='gpu')
	parser.add_argument('--save-path', dest='save_path', default='', type=str, help='save downsampled state features')

	args = parser.parse_args()

	if args.tag=='':
		vis = visdom.Visdom()
	else:
		vis = visdom.Visdom(env=args.tag)

	'''global st, ed
	st = 1386+1
	ed = 1425+1'''

	path = args.path
	save_path = ''
	if args.save_path!='':
		save_path = os.path.join(os.path.dirname(path), args.save_path)

	plot_3d = False
	if args.dim==3:
		plot_3d = True

	if args.states:
		buff = np.load(args.path)[:100000]
		states	= np.squeeze(np.array([elt[0] for elt in buff]))
		vis_states(vis, states, plot_3d = plot_3d, save_path = save_path)

	if args.actions:
		buff = np.load(args.path)
		actions	= np.squeeze(np.array([elt[1] for elt in buff]))
		vis_actions(vis, actions)

	if args.rewards:
		buff = np.load(args.path)
		rewards = np.squeeze(np.array([elt[2] for elt in buff]))
		vis_rewards(vis, rewards, mode='line')

	if args.next_states:
		buff = np.load(args.path)
		next_states = np.squeeze(np.array([elt[3] for elt in buff]))
		vis_states(vis, next_states, plot_3d = plot_3d, save_path = save_path)

	if args.dones:
		buff = np.load(args.path)
		dones	= np.squeeze(np.array([elt[4] for elt in buff]))
		vis_dones(vis, dones)

	if args.qvalues:
		buff = np.load(args.path)
		if args.tseries:
			#vis_policyloss(vis, args.exp_path, buff, None, True, args.gpu)
			vis_exp_qvalues(args, vis, args.exp_path, buff, None, True, args.gpu)
		else:
			filename = 'sac_weights6750000.pkl'
			if not args.model_name=='':
				filename = args.model_name
			vis_qvalues2(vis, args.exp_path, buff, filename, True, args.gpu)

	if args.val:
		vis_validation(vis, os.path.dirname(path))

from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import numpy as np
import random
from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
import gc

class Custom_ReplayBuffer(ReplayBuffer):
    '''
    Custom ReplayBuffer with addition of info data and additional methods

    '''
    def __init__(self, size):
        super(Custom_ReplayBuffer, self).__init__(size)
        self._done_idx = []
        self._num_termination_states = 12
        self._termination_state_idx = [[] for i in range(self._num_termination_states)]
        self._max_time_to_termination = 3
        self._time_to_termination_idx = [[] for i in range(self._max_time_to_termination)]

    # def __init__(self, size, storage, next_idx):
    #     super(Custom_ReplayBuffer, self).__init__(size)
    #     self._done_idx = []
    #     self._storage, self._next_idx = storage, next_idx


    def add_old(self, obs_t, action, reward, obs_tp1, done, info):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        :param info: (int) episode termination code
        """
        data = (obs_t, action, reward, obs_tp1, done, info)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def add(self, obs_t, action, reward, obs_tp1, done, term_state, time_to_termination):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        :param term_state: (int) episode termination code
        :param time_to_termination: (int) time to terminate episode, eg. t=0 for done case.
        """
        data = (obs_t, action, reward, obs_tp1, done, term_state, time_to_termination)

        data_popped = None
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            data_popped = self._storage[self._next_idx]
            self._storage[self._next_idx] = data
        
        # Remove from index lists first
        if data_popped is not None:
            (obs_t_p, action_p, reward_p, obs_tp1_p, done_p, term_state_p, time_to_termination_p) = data_popped
            if done_p:
                self._done_idx.remove(self._next_idx)
                self._termination_state_idx[term_state_p].remove(self._next_idx)

            
            if time_to_termination_p < self._max_time_to_termination:
                self._time_to_termination_idx[time_to_termination_p].remove(self._next_idx)

            del data_popped
            del obs_t_p
            del obs_tp1_p
            gc.collect()

        # Add to index lists
        if done:
            self._done_idx.append(self._next_idx)
            if term_state >= 0:
                self._termination_state_idx[term_state].append(self._next_idx)
            else:
                print("Some error in term_state_code in done condition.")

        if time_to_termination < self._max_time_to_termination:
            self._time_to_termination_idx[time_to_termination].append(self._next_idx)

        
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample_old(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, infos = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, info = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            infos.append(info)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(infos)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, term_states, time_to_terminations = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, term_state, time_to_termination = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            term_states.append(term_state)
            time_to_terminations.append(time_to_termination)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(term_states), np.array(time_to_terminations)


    def _compute_storage_done_idx(self):
        n = len(self._storage)
        self._done_idx = [i for i in range(n) if self._storage[i][4] == 1]

    def sample_done(self, batch_size, **_kwargs):

        if len(self._done_idx) == 0:
            self._compute_storage_done_idx()
        
        idxes = random.sample(self._done_idx, batch_size)
        return self._encode_sample(idxes)

    def sample_done_term_state(self, batch_size, termination_state_list):

        final_list = []
        if len(termination_state_list) == 1:
            final_list = self._termination_state_idx[termination_state_list[0]]
        
        else:
            for term_state in termination_state_list:
                final_list += self._termination_state_idx[term_state]

        
        idxes = random.sample(final_list, batch_size)
        return self._encode_sample(idxes)
    
    def sample_time_to_termination(self, batch_size, time_to_termination_list):

        final_list = []
        if len(time_to_termination_list) == 1:
            final_list = self._time_to_termination_idx[time_to_termination_list[0]]
        
        else:
            for time_to_termination in time_to_termination_list:
                final_list += self._time_to_termination_idx[time_to_termination]

        
        idxes = random.sample(final_list, batch_size)
        return self._encode_sample(idxes)

    def sample_random_episode(self, batch_size):

        '''
        Sample random episode method following Episodic Backward Update (EBU) paper.
        Code is based on EBU codebase.
        '''

        terminal_array = self._done_idx

        batchnum = 0
        while batchnum == 0:
            # exclude some early and final episodes from sampling due to indexing issues,
            # sample two episodes (ind1 for main, and ind2 for the remaining steps to make multiple of 32)
            ind = random.sample(range(5,len(terminal_array)-3), 2)
            ind1 = ind[0]
            ind2 = ind[1]

            # NOTE: Custom change: Removed +3 from terminal_array[ind1-1]+3 to include complete episode
            # Perhaps EBU code had a stack of 3 frames, so it had +3 in it.
            indice_array = range(terminal_array[ind1],terminal_array[ind1-1],-1)
            epi_len = len(indice_array)
            batchnum = int(np.ceil(epi_len/float(batch_size)))

        remainindex = int(batchnum * batch_size - epi_len)

        # Normally an episode does not have steps=multiple of 32.
        # Fill last minibatch with redundant steps from another episode
        indice_array= np.append(indice_array, range(terminal_array[ind2], terminal_array[ind2]-remainindex, -1))
        indice_array = indice_array.astype(int)
        idxes = list(indice_array)

        return batchnum, self._encode_sample(idxes)

class Custom_PrioritizedReplayBuffer(Custom_ReplayBuffer):

    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(Custom_PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done, term_state, time_to_termination):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        :param info: (int) episode termination code
        """
        data = (obs_t, action, reward, obs_tp1, done, term_state, time_to_termination)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
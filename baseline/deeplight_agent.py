

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.merge import concatenate, add

import pickle

### network_agent

import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
import random
from keras.engine.topology import Layer
import os

from agent import Agent

class Selector(Layer):

    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.select_neuron = K.constant(value=self.select)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.cast(K.equal(x, self.select_neuron), dtype="float32")

    def get_config(self):
        config = {"select": self.select}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def conv2d_bn(input_layer, index_layer,
              filters=16,
              kernel_size=(3, 3),
              strides=(1, 1)):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  name="conv{0}".format(index_layer))(input_layer)
    bn = BatchNormalization(axis=bn_axis, scale=False, name="bn{0}".format(index_layer))(conv)
    act = Activation('relu', name="act{0}".format(index_layer))(bn)
    pooling = MaxPooling2D(pool_size=2)(act)
    x = Dropout(0.3)(pooling)
    return x


class NetworkAgent(Agent):

    @staticmethod
    def _unison_shuffled_copies(Xs, Y, sample_weight):
        p = np.random.permutation(len(Y))
        new_Xs = []
        for x in Xs:
            assert len(x) == len(Y)
            new_Xs.append(x[p])
        return new_Xs, Y[p], sample_weight[p]

    @staticmethod
    def _cnn_network_structure(img_features):
        conv1 = conv2d_bn(img_features, 1, filters=32, kernel_size=(8, 8), strides=(4, 4))
        conv2 = conv2d_bn(conv1, 2, filters=16, kernel_size=(4, 4), strides=(2, 2))
        img_flatten = Flatten()(conv2)
        return img_flatten

    @staticmethod
    def _shared_network_structure(state_features, dense_d):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_shared_1")(state_features)
        return hidden_1

    @staticmethod
    def _separate_network_structure(state_features, dense_d, num_actions, memo=""):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_separate_branch_{0}_1".format(memo))(state_features)
        q_values = Dense(num_actions, activation="linear", name="q_values_separate_branch_{0}".format(memo))(hidden_1)
        return q_values

    def load_model(self, file_name, path = None):

        if not path:
            path = self.dic_path["PATH_TO_MODEL"]

        self.q_network = load_model(os.path.join(os.getcwd(), path, "%s_q_network.h5" % file_name), custom_objects={"Selector": Selector})
        print("succeed in loading model %s"%file_name)

    def load_qbar_model(self, file_name, path = None):

        if not path:
            path = self.dic_path["PATH_TO_MODEL"]


        self.q_network_bar = load_model(os.path.join(os.getcwd(), path, "%s_q_network.h5" % file_name), custom_objects={"Selector": Selector})
        print("succeed in loading model %s as q_network_bar"%file_name)



    def save_model(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_q_network.h5" % file_name))


    def choose(self, count, if_pretrain):

        ''' choose the best action for current state '''

        q_values = self.q_network.predict(self.convert_state_to_input(self.state))
        # print(q_values)
        if if_pretrain:
            self.action = np.argmax(q_values[0])
        else:
            if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
                self.action = random.randrange(len(q_values[0]))
                print("##Explore")
            else:  # exploitation
                self.action = np.argmax(q_values[0])
            if self.dic_agent_conf["EPSILON"] > 0.001 and count >= 20000:
                self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["EPSILON"] * 0.9999
        return self.action, q_values

    def choose_action(self, count, state):

        ''' choose the best action for current state '''

        #q_values = self.q_network.predict(self.convert_state_to_input(state))
        state = [[state[feature]] for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]

        q_values = self.q_network.predict(state)
        if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
            self.action = random.randrange(len(q_values[0]))
        else:  # exploitation
            self.action = np.argmax(q_values[0])

        #if self.dic_agent_conf["EPSILON"] > 0.001 and count >= 600:
        #    self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["EPSILON"] * 0.99
        return self.action

    def build_memory(self):

        return []

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''

        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"Selector": Selector})
        network.set_weights(network_weights)
        network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss="mean_squared_error")
        return network

    #
    # def batch_predict(self,file_name="temp"):
    #     f_samples = open("./records/DQN_v1/" + file_name[:file_name.rfind("_")] + "predict_pretrain.txt", "a")
    #     f_samples_head = ["state.cur_phase", "state.time_this_phase",
    #                       "target",
    #                       "action",
    #                       "reward"]
    #     f_samples.write('\t'.join(f_samples_head) + "\n")
    #     len_memory = len(self.memory)
    #     for i in range(len_memory):
    #         state, action, reward, next_state = self.memory[i]
    #         q_values = self.q_network.predict(self.convert_state_to_input(state))
    #         f_samples.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
    #             str(state.cur_phase[0]), str(state.time_this_phase), str(q_values),
    #             str(action), str(reward)
    #         ))

    def remember(self, state, action, reward, next_state):

        ''' log the history '''
        self.memory.append([state, action, reward, next_state])

    def forget(self):

        ''' remove the old history if the memory is too large '''

        if len(self.memory) > self.dic_agent_conf["MAX_MEMORY_LEN"]:
            print("length of memory: {0}, before forget".format(len(self.memory)))
            self.memory = self.memory[-self.dic_agent_conf["MAX_MEMORY_LEN"]:]
            print("length of memory: {0}, after forget".format(len(self.memory)))

    def _get_next_estimated_reward(self, next_state):

        if self.dic_agent_conf["DDQN"]:
            a_max = np.argmax(self.q_network.predict(
                self.convert_state_to_input(next_state))[0])
            next_estimated_reward = self.q_network_bar.predict(
                self.convert_state_to_input(next_state))[0][a_max]
            return next_estimated_reward
        else:
            next_state = [[next_state[feature]] for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
            next_estimated_reward = np.max(self.q_network_bar.predict(next_state))
            return next_estimated_reward

    def update_network_bar(self):

        ''' update Q bar '''

        if self.q_bar_outdated >= self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]:
            self.q_network_bar = self.build_network_from_copy(self.q_network)
            self.q_bar_outdated = 0


MEMO = "Deeplight"


class DeeplightAgent(NetworkAgent):



    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round=None, best_round=None):

        super(DeeplightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path)

        #self.num_actions = self.dic_sumo_env_conf["ACTION_DIM"]
        #self.num_phases = 2
        self.num_actions = len(self.dic_sumo_env_conf["PHASE"])
        self.num_phases = len(self.dic_sumo_env_conf["PHASE"])
        self.num_lanes = np.sum(np.array(list(self.dic_sumo_env_conf["LANE_NUM"].values())))

        if cnt_round:
            try:
                self.load_model("round_" + str(cnt_round-1))
            except:
                print("fail to load model %s as q_network_bar" %("round_"+str(cnt_round-1)))
        else:
            self.q_network = self.build_network()
        #self.save_model("init_model")
        self.update_outdated = 0

        self.q_network_bar = self.build_network_from_copy(self.q_network)
        self.q_bar_outdated = 0
        if not self.dic_agent_conf["SEPARATE_MEMORY"]:
            self.memory = self.build_memory()
        else:
            self.memory = self.build_memory_separate()
        self.average_reward = None



    def reset_update_count(self):

        self.update_outdated = 0
        self.q_bar_outdated = 0

    def set_update_outdated(self):

        self.update_outdated = - 2*self.dic_agent_conf["UPDATE_PERIOD"]
        self.q_bar_outdated = 2*self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]

    def convert_state_to_input(self, state):

        ''' convert a state struct to the format for neural network input'''

        return [state[feature_name]
                for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]

    def build_network(self):

        '''Initialize a Q network'''

        # initialize feature node
        #dic_input_node = {}
        #for feature_name in self.dic_sumo_env_conf["LIST_STATE_FEATURE"]:
        #    dic_input_node[feature_name] = Input(shape=self.dic_sumo_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()],
        #                                             name="input_"+feature_name)
        dic_input_node = {}
        for feature_name in self.dic_sumo_env_conf["LIST_STATE_FEATURE"]:
            if "phase" in feature_name:
                _shape = self.dic_sumo_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()]
            else:
                _shape = (self.dic_sumo_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()][0] * self.num_lanes,)
            dic_input_node[feature_name] = Input(shape=_shape,
                                                 name="input_" + feature_name)

        # add cnn to image features
        dic_flatten_node = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if len(self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]) > 1:
                dic_flatten_node[feature_name] = self._cnn_network_structure(dic_input_node[feature_name])
            else:
                dic_flatten_node[feature_name] = dic_input_node[feature_name]

        # concatenate features
        list_all_flatten_feature = []
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            list_all_flatten_feature.append(dic_flatten_node[feature_name])
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")

        # shared dense layer
        shared_dense = self._shared_network_structure(all_flatten_feature, self.dic_agent_conf["D_DENSE"])

        # build phase selector layer
        if "cur_phase" in self.dic_traffic_env_conf["LIST_STATE_FEATURE"] and self.dic_agent_conf["PHASE_SELECTOR"]:
            list_selected_q_values = []
            for phase in range(self.num_phases):
                locals()["q_values_{0}".format(phase)] = self._separate_network_structure(
                    shared_dense, self.dic_agent_conf["D_DENSE"], self.num_actions, memo=phase)
                locals()["selector_{0}".format(phase)] = Selector(
                    phase, name="selector_{0}".format(phase))(dic_input_node["cur_phase"])
                locals()["q_values_{0}_selected".format(phase)] = Multiply(name="multiply_{0}".format(phase))(
                    [locals()["q_values_{0}".format(phase)],
                     locals()["selector_{0}".format(phase)]]
                )
                list_selected_q_values.append(locals()["q_values_{0}_selected".format(phase)])
            q_values = Add()(list_selected_q_values)
        else:
            q_values = self._separate_network_structure(shared_dense, self.dic_agent_conf["D_DENSE"], self.num_actions)

        network = Model(inputs=[dic_input_node[feature_name]
                                for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]],
                        outputs=q_values)
        network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss="mean_squared_error")
        network.summary()

        return network

    def build_memory_separate(self):
        memory_list=[]
        for i in range(self.num_phases):
            memory_list.append([[] for j in range(self.num_actions)])
        return memory_list

    def remember(self, state, action, reward, next_state):

        if self.dic_agent_conf["SEPARATE_MEMORY"]:
            ''' log the history separately '''
            self.memory[state["cur_phase"][0]][action].append([state, action, reward, next_state])
        else:
            self.memory.append([state, action, reward, next_state])

    def forget(self, if_pretrain):

        max_keep_size = int(self.dic_agent_conf["KEEP_OLD_MEMORY"] * self.dic_agent_conf["MAX_MEMORY_LEN"])

        if self.dic_agent_conf["SEPARATE_MEMORY"]:
            ''' remove the old history if the memory is too large, in a separate way '''
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    if if_pretrain:
                        random.shuffle(self.memory[phase_i][action_i])
                    if len(self.memory[phase_i][action_i]) > self.dic_agent_conf["MAX_MEMORY_LEN"]:
                        print("length of memory (state {0}, action {1}): {2}, before forget".format(
                            phase_i, action_i, len(self.memory[phase_i][action_i])))
                        keep_size = min(max_keep_size, len(self.memory[phase_i][action_i]))
                        remain_size = self.dic_agent_conf["MAX_MEMORY_LEN"] - keep_size
                        self.memory[phase_i][action_i] = \
                            self.memory[phase_i][action_i][:keep_size] + \
                            self.memory[phase_i][action_i][-remain_size:]
                    print("length of memory (state {0}, action {1}): {2}, after forget".format(
                        phase_i, action_i, len(self.memory[phase_i][action_i])))
        else:
            if len(self.memory) > self.dic_agent_conf["MAX_MEMORY_LEN"]:
                print("length of memory: {0}, before forget".format(len(self.memory)))
                keep_size = min(max_keep_size, len(self.memory))
                remain_size = self.dic_agent_conf["MAX_MEMORY_LEN"] - keep_size
                self.memory = \
                    self.memory[:keep_size] + \
                    self.memory[-remain_size:]
            print("length of memory: {0}, after forget".format(len(self.memory)))

    def _cal_average(self, sample_memory):

        list_reward = []
        average_reward = np.zeros((self.num_phases, self.num_actions))
        for phase_i in range(self.num_phases):
            list_reward.append([])
            for action_i in range(self.num_actions):
                list_reward[phase_i].append([])
        for [state, action, reward, _] in sample_memory:
            phase = state["cur_phase"][0]
            list_reward[phase][action].append(reward)

        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                if len(list_reward[phase_i][action_i]) != 0:
                    average_reward[phase_i][action_i] = np.average(list_reward[phase_i][action_i])

        return average_reward

    def _cal_average_separate(self,sample_memory):
        ''' Calculate average rewards for different cases '''

        average_reward = np.zeros((self.num_phases, self.num_actions))
        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                len_sample_memory = len(sample_memory[phase_i][action_i])
                if len_sample_memory > 0:
                    list_reward = []
                    for i in range(len_sample_memory):
                        state, action, reward, _ = sample_memory[phase_i][action_i][i]
                        list_reward.append(reward)
                    average_reward[phase_i][action_i]=np.average(list_reward)
        return average_reward

    def get_sample(self, memory_slice, dic_state_feature_arrays, Y, gamma, prefix, use_average):

        len_memory_slice = len(memory_slice)

        #f_samples = open(os.path.join(self.path_set.PATH_TO_OUTPUT, "{0}_memory".format(prefix)), "a")

        for i in range(len_memory_slice):
            state, action, reward, next_state = memory_slice[i]
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                dic_state_feature_arrays[feature_name].append(state[feature_name])


            next_estimated_reward = self._get_next_estimated_reward(next_state)

            total_reward = reward + gamma * next_estimated_reward
            if not use_average:
                state = [[state[feature]] for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
                target = self.q_network.predict(state)
            else:
                target = np.copy(np.array([self.average_reward[state["cur_phase"][0]]]))

            pre_target = np.copy(target)
            target[0][action] = total_reward
            Y.append(target[0])

            # == don't log the samples when updating the network
            #for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            #    if "map" not in feature_name:
            #        f_samples.write("{0}\t".format(str(getattr(state, feature_name))))
            #f_samples.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
            #    str(pre_target), str(target),
            #    str(action), str(reward), str(next_estimated_reward)
            #))
        #f_samples.close()

        return dic_state_feature_arrays, Y

    def train_network(self, Xs, Y, prefix, if_pretrain):

        if if_pretrain:
            epochs = self.dic_agent_conf["EPOCHS_PRETRAIN"]
        else:
            epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        hist = self.q_network.fit(Xs, Y, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=2, validation_split=0.3, callbacks=[early_stopping])
        self.save_model(prefix)

    def update_network(self, if_pretrain, use_average, current_time):

        ''' update Q network '''

        if current_time - self.update_outdated < self.dic_agent_conf["UPDATE_PERIOD"]:
            return

        self.update_outdated = current_time

        # prepare the samples
        if if_pretrain:
            gamma = self.dic_agent_conf["GAMMA_PRETRAIN"]
            print("precision ", K.floatx())
        else:
            gamma = self.dic_agent_conf["GAMMA"]

        dic_state_feature_arrays = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        # get average state-action reward
        if self.dic_agent_conf["SEPARATE_MEMORY"]:
            self.average_reward = self._cal_average_separate(self.memory)
        else:
            self.average_reward = self._cal_average(self.memory)

        # ================ sample memory ====================
        if self.dic_agent_conf["SEPARATE_MEMORY"]:
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    sampled_memory = self._sample_memory(
                        gamma=gamma,
                        with_priority=self.dic_agent_conf["PRIORITY_SAMPLING"],
                        memory=self.memory[phase_i][action_i],
                        if_pretrain=if_pretrain)
                    dic_state_feature_arrays, Y = self.get_sample(
                        sampled_memory, dic_state_feature_arrays, Y, gamma, current_time, use_average)
        else:
            sampled_memory = self._sample_memory(
                gamma=gamma,
                with_priority=self.dic_agent_conf["PRIORITY_SAMPLING"],
                memory=self.memory,
                if_pretrain=if_pretrain)
            dic_state_feature_arrays, Y = self.get_sample(
                sampled_memory, dic_state_feature_arrays, Y, gamma, current_time, use_average)
        # ================ sample memory ====================

        Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
        Y = np.array(Y)
        sample_weight = np.ones(len(Y))
        # shuffle the training samples, especially for different phases and actions
        Xs, Y, _ = self._unison_shuffled_copies(Xs, Y, sample_weight)

        if if_pretrain:
            pickle.dump(Xs, open(os.path.join(self.path_set.PATH_TO_OUTPUT, "Xs.pkl"), "wb"))
            pickle.dump(Y, open(os.path.join(self.path_set.PATH_TO_OUTPUT, "Y.pkl"), "wb"))
        # ============================  training  =======================================

        self.train_network(Xs, Y, current_time, if_pretrain)
        self.q_bar_outdated += 1
        self.forget(if_pretrain=if_pretrain)

    def _sample_memory(self, gamma, with_priority, memory, if_pretrain):

        len_memory = len(memory)

        if not if_pretrain:
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len_memory)
        else:
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE_PRETRAIN"], len_memory)

        if with_priority:
            # sample with priority
            sample_weight = []
            for i in range(len_memory):
                state, action, reward, next_state = memory[i]


                next_estimated_reward = self._get_next_estimated_reward(next_state)

                total_reward = reward + gamma * next_estimated_reward
                target = self.q_network.predict(
                    self.convert_state_to_input(state))
                pre_target = np.copy(target)
                target[0][action] = total_reward

                # get the bias of current prediction
                weight = abs(pre_target[0][action] - total_reward)
                sample_weight.append(weight)

            priority = self._cal_priority(sample_weight)
            p = random.choices(range(len(priority)), weights=priority, k=sample_size)
            sampled_memory = np.array(memory)[p]
        else:
            sampled_memory = random.sample(memory, sample_size)

        return sampled_memory

    @staticmethod
    def _cal_priority(sample_weight):
        pos_constant = 0.0001
        alpha = 1
        sample_weight_np = np.array(sample_weight)
        sample_weight_np = np.power(sample_weight_np + pos_constant, alpha) / sample_weight_np.sum()
        return sample_weight_np

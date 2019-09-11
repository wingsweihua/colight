


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
        return K.cast(K.equal(x, self.select_neuron), dtype="float64")

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

    def load_network(self, file_name):
        self.q_network = load_model(os.path.join(os.getcwd(), self.dic_path["PATH_TO_MODEL"], "{0}.h5".format(file_name)), custom_objects={"Selector": Selector})
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name):
        self.q_network_bar = load_model(os.path.join(os.getcwd(), self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name), custom_objects={"Selector": Selector})
        print("succeed in loading model %s"%file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def prepare_Xs_Y(self, sample_set):

        NORMALIZATION_FACTOR = 20

        # forget
        ind_end = len(sample_set)
        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        sample_set = sample_set[ind_sta: ind_end]

        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(sample_set))
        print("memory samples number:", sample_size)

        sample_slice = random.sample(sample_set, sample_size)

        dic_state_feature_arrays = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        for i in range(len(sample_slice)):
            state, action, next_state, reward, instant_reward, _ = sample_slice[i]

            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                dic_state_feature_arrays[feature_name].append(state[feature_name])

            _state = []
            _next_state = []
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                _state.append([state[feature_name]])
                _next_state.append([next_state[feature_name]])
            target = self.q_network.predict(_state)

            next_state_qvalues = self.q_network_bar.predict(_next_state)

            if self.dic_agent_conf["LOSS_FUNCTION"] == "mean_squared_error":
                final_target = np.copy(target[0])
                final_target[action] = reward/NORMALIZATION_FACTOR + self.dic_agent_conf["GAMMA"] * next_state_qvalues[0][action]
            elif self.dic_agent_conf["LOSS_FUNCTION"] == "categorical_crossentropy":
                raise NotImplementedError

            Y.append(final_target)

        self.Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                   self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
        self.Y = np.array(Y)


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
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        return network
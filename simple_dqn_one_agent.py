
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.merge import concatenate, add
import random
import os
import pickle

from network_agent import NetworkAgent, conv2d_bn, Selector
import json


class SimpleDQNOneAgent(NetworkAgent):

    def build_network(self):

        '''Initialize a Q network'''

        # initialize feature node
        dic_input_node = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "phase" in feature_name or "adjacency" in feature_name or "pressure" in feature_name:
                _shape = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
            else:
                _shape = (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*self.num_lanes,)
                # _shape = (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()])
            dic_input_node[feature_name] = Input(shape=_shape,
                                                     name="input_"+feature_name)

        # add cnn to image features
        dic_flatten_node = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if len(self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]) > 1:
                dic_flatten_node[feature_name] = Flatten()(dic_input_node[feature_name])
            else:
                dic_flatten_node[feature_name] = dic_input_node[feature_name]

        # concatenate features
        list_all_flatten_feature = []
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            list_all_flatten_feature.append(dic_flatten_node[feature_name])
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")

        # shared dense layer, N_LAYER
        locals()["dense_0"] = Dense(self.dic_agent_conf["D_DENSE"], activation="relu", name="dense_0")(all_flatten_feature)
        for i in range(1, self.dic_agent_conf["N_LAYER"]):
            locals()["dense_%d"%i] = Dense(self.dic_agent_conf["D_DENSE"], activation="relu", name="dense_%d"%i)(locals()["dense_%d"%(i-1)])
        q_values = Dense(self.num_actions, activation="linear", name="q_values")(locals()["dense_%d"%(self.dic_agent_conf["N_LAYER"]-1)])
        network = Model(inputs=[dic_input_node[feature_name]
                                for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]],
                        outputs=q_values)
        network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.summary()

        return network

    def convert_state_to_input(self, s):
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            inputs = []
            for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature == "cur_phase":
                    inputs.append(np.array([self.dic_traffic_env_conf['PHASE']
                                            [self.dic_traffic_env_conf['SIMULATOR_TYPE']][s[feature][0]]]))
                else:
                    inputs.append(np.array([s[feature]]))
            return inputs
        else:
            return [np.array([s[feature]]) for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]



    def choose_action(self, count, states):
        '''
        choose the best action for current state
        -input: state:[[state inter1],[state inter1]]
        -output: act: [#agents,num_actions]
        '''
        dic_state_feature_arrays = {} # {feature1: [inter1, inter2,..], feature2: [inter1, inter 2...]}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []


        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                # print(s[feature_name])
                if "cur_phase" in feature_name:
                    dic_state_feature_arrays[feature_name].append(np.array(self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']][s[feature_name][0]]))
                else:
                    dic_state_feature_arrays[feature_name].append(np.array(s[feature_name]))

        state_input = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                       self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]

        # print("----------State Input",state_input)
        # print(dic_state_feature_arrays)

        q_values = self.q_network.predict(state_input)
        if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
            action = np.random.randint(len(q_values[0]), size=len(q_values))
        else:  # exploitation
            action = np.argmax(q_values, axis=1)

        return action

    def choose_action_separate(self, count, states):
        '''
        choose the best action for current state
        -input: state:[[state inter1],[state inter1]]
        -output: act: [#agents,num_actions]
        '''

        actions = []

        for state in states:
            action = self.choose_action_i(count, state)
            actions.append(action)

        return actions

    def choose_action_i(self, count, state):

        ''' choose the best action for current state '''
        state_input = self.convert_state_to_input(state)
        q_values = self.q_network.predict(state_input)
        if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
            action = random.randrange(len(q_values[0]))
        else:  # exploitation
            action = np.argmax(q_values[0])

        return action

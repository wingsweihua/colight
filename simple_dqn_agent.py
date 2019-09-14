
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


class SimpleDQNAgent(NetworkAgent): 

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

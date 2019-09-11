
import pickle
from network_agent import NetworkAgent, Selector

import numpy as np
from keras.layers import Input,  Multiply, Add
from keras.models import Model
from keras.optimizers import RMSprop

from keras.layers.merge import concatenate


class LitAgent(NetworkAgent): 
    def build_network(self):

        '''Initialize a Q network'''
        dic_input_node = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "phase" in feature_name and self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                _shape = (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()][0],)
                d_phase_encoding = _shape[0]
            elif "phase" in feature_name and not self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                _shape = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
                d_phase_encoding = _shape[0]
            else:
                _shape = (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()][0] * self.num_lanes,)
            print("_shape", _shape, feature_name)
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
            for phase_id in range(1, self.num_phases+1):
                if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                    # print('d_phase_encoding:',d_phase_encoding)#d_phase_encoding: 96
                    if d_phase_encoding == 4:
                        phase_expansion = self.dic_traffic_env_conf["phase_expansion_4_lane"][phase_id]
                    elif d_phase_encoding == 8:
                        phase_expansion = self.dic_traffic_env_conf["phase_expansion"][phase_id]
                    else:
                        raise NotImplementedError
                else:
                    phase_expansion = phase_id
                locals()["q_values_{0}".format(phase_id)] = self._separate_network_structure(
                    shared_dense, self.dic_agent_conf["D_DENSE"], self.num_actions, memo=phase_id)
                locals()["selector_{0}".format(phase_id)] = Selector(
                    phase_expansion, d_phase_encoding=d_phase_encoding, d_action=self.num_actions,
                    name="selector_{0}".format(phase_id))(dic_input_node["cur_phase"])
                locals()["q_values_{0}_selected".format(phase_id)] = Multiply(name="multiply_{0}".format(phase_id))(
                    [locals()["q_values_{0}".format(phase_id)],
                     locals()["selector_{0}".format(phase_id)]]
                )
                list_selected_q_values.append(locals()["q_values_{0}_selected".format(phase_id)])
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

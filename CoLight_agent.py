import numpy as np 
import os
import copy
from agent import Agent
import random 
import time
"""
Model for CoLight in paper "CoLight: Learning Network-level Cooperation for Traffic Signal
Control", in submission. 
"""
import keras
from keras import backend as K
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model, model_from_json, load_model
from keras.layers.merge import concatenate, add, dot, maximum, multiply
from keras.layers.core import Activation
from keras.utils import np_utils,to_categorical
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping, TensorBoard
from frap import slice_tensor, relation

SEED=6666
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

use_lane_ind = []  # all lanes to be used instead of right lanes
for ii in range(8):
    if ii % 2 == 0:
        use_lane_ind.append(ii // 2 * 3)
    else:
        use_lane_ind.append(ii // 2 * 3 + 1)


def slice_icap(x, name=None):
    if name == 'phase':
        return x[:, :, :8]
    elif name == 'num_vehicle':
        return x[:, :, 8:]
    else:
        print('name not included')


def slice_icap_2(x, index):
    return x[:, index, :]


def slice_tensor_2(x, index):
    x_shape = K.int_shape(x)
    if len(x_shape) == 4:
        return x[:, :, index, :]
    elif len(x_shape) == 3:
        return x[:, :, index]


def slice_tensor_3(x, index):
    x_shape = K.int_shape(x)
    if len(x_shape) == 5:
        return x[:, index, :, :, :]
    elif len(x_shape) == 3:
        return x[:, index, :]


def slice_tensor_4(x, index):
    for ind in index:
        if ind == 0:
            tensor = Reshape((1,))(x[:, ind])
        else:
            tensor = concatenate([tensor, Reshape((1,))(x[:, ind])])
    return tensor


rotation_matrix_4_p = np.zeros((8, 8))
j = 4
for i in range(8):
    rotation_matrix_4_p[i][j] = 1
    j = (j + 1) % 8


class RepeatVector3D(Layer):
    def __init__(self,times,**kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def call(self, inputs):
        # [batch,agent,dim]->[batch,1,agent,dim]
        # [batch,1,agent,dim]->[batch,agent,agent,dim]

        return K.tile(K.expand_dims(inputs, 1), [1, self.times, 1, 1])

    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CoLightAgent(Agent): 
    def __init__(
            self,
            dic_agent_conf=None,
            dic_traffic_env_conf=None,
            dic_path=None,
            cnt_round=None,
            best_round=None, bar_round=None,intersection_id="0"
    ):
        """
        #1. compute the (dynamic) static Adjacency matrix, compute for each state
        -2. #neighbors: 5 (1 itself + W,E,S,N directions)
        -3. compute len_features
        -4. self.num_actions
        """
        super(CoLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,intersection_id)

        self.att_regulatization = dic_agent_conf['att_regularization']
        self.CNN_layers = dic_agent_conf['CNN_layers']
        
        # TODO: n_agents should pass as parameter
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)
        self.vec=np.zeros((1, self.num_neighbors))
        self.vec[0][0] = 1

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.len_feature = self.compute_len_feature()
        self.memory = self.build_memory()

        if cnt_round == 0: 
            # initialization
            self.q_network = self.build_network()
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.q_network.load_weights(
                    os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(intersection_id)), 
                    by_name=True)
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                if best_round:
                    # use model pool
                    self.load_network("round_{0}_inter_{1}".format(best_round,self.intersection_id))

                    if bar_round and bar_round != best_round and cnt_round > 10:
                        # load q_bar network from model pool
                        self.load_network_bar("round_{0}_inter_{1}".format(bar_round,self.intersection_id))
                    else:
                        if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                            if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                                self.load_network_bar("round_{0}".format(
                                    max((best_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                    self.intersection_id))
                            else:
                                self.load_network_bar("round_{0}_inter_{1}".format(
                                    max(best_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                    self.intersection_id))
                        else:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max(best_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))

                else:
                    # not use model pool
                    # print('init q load')
                    self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
                    # print('init q_bar load')
                    if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                        if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                self.intersection_id))
                        else:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        # decay the epsilon
        """
        "EPSILON": 0.8,
        "EPSILON_DECAY": 0.95,
        "MIN_EPSILON": 0.2,
        """
        if os.path.exists(
            os.path.join(
                self.dic_path["PATH_TO_MODEL"], 
                "round_-1_inter_{0}.h5".format(intersection_id))):
            # the 0-th model is pretrained model
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f'%(cnt_round,self.dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def compute_len_feature(self):
        from functools import reduce
        len_feature=tuple()
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "adjacency" in feature_name:
                continue
            elif "phase" in feature_name:
                len_feature += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
            elif feature_name=="lane_num_vehicle":
                len_feature += (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*self.num_lanes,)
            elif feature_name == "pressure_of_movement":
                len_feature += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()]
        return sum(len_feature)

    """
    components of the network
    1. MLP encoder of features
    2. CNN layers
    3. q network
    """
    def MLP(self, In_0, layers=[128, 128]):
        """
        Currently, the MLP layer 
        -input: [batch,#agents,feature_dim]
        -outpout: [batch,#agents,128]
        """
        # In_0 = Input(shape=[self.num_agents,self.len_feature])
        for layer_index, layer_size in enumerate(layers):
            if layer_index == 0:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal', name='Dense_embed_%d' % layer_index)(In_0)
            else:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal', name='Dense_embed_%d' % layer_index)(h)

        return h

    def iCAP(self, In, MLP_layers):
        """
        embed iCAP into Colight, process each intersection separately
        """
        phase_input = Lambda(slice_icap, arguments={'name': 'phase'}, output_shape=[8], name="inter_phase")(In[0])  # None, 9, 8
        num_vehicle_input = Lambda(slice_icap, arguments={'name': 'num_vehicle'}, output_shape=[8], name="inter_num_vehicle")(In[0])
        p = Activation('sigmoid')(Embedding(2, 4, input_length=8)(phase_input))
        d = Dense(4, activation="sigmoid", name="num_vec_mapping")
        lane_embedding = Dense(16, activation="relu", name="lane_embedding")
        relation_embed = Embedding(2, 4, name="relation_embedding")
        _lane_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu", name="lane_conv")
        _relation_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                                name="relation_conv")
        _combine_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                               name="combine_conv")
        _before_merge = Conv2D(1, kernel_size=(1, 1), activation="linear", name="befor_merge")

        for slice_ind in range(self.num_agents):
            phase_slice = Lambda(slice_icap_2, arguments={'index': slice_ind}, output_shape=[8, ] + [4],
                                 name="inter_phase_%d" % slice_ind)(p)  # make this None 8 4
            num_vehicle_slice = Lambda(slice_icap_2, arguments={'index': slice_ind}, output_shape=[8],
                                       name="inter_num_vehicle_%d" % slice_ind)(num_vehicle_input)  # make this None 8
            dic_lane = {}
            for i, m in enumerate(self.dic_traffic_env_conf["list_lane_order"]):
                tmp_vec = d(
                    Lambda(slice_tensor, arguments={"index": i}, name="vec_%d_%d" % (i, slice_ind))(num_vehicle_slice))
                tmp_phase = Lambda(slice_tensor, arguments={"index": i}, name="phase_%d_%d" % (i, slice_ind))(
                    phase_slice)
                dic_lane[m] = concatenate([tmp_vec, tmp_phase], name="lane_%d_%d" % (i, slice_ind))

            list_phase_pressure = []

            for phase in self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']]:
                m1, m2 = phase.split("_")
                list_phase_pressure.append(
                    add([lane_embedding(dic_lane[m1]), lane_embedding(dic_lane[m2])],
                        name=phase + '_' + str(slice_ind)))

            constant = Lambda(relation, arguments={"dic_traffic_env_conf": self.dic_traffic_env_conf},
                              name="constant_%d" % slice_ind)(num_vehicle_slice)
            relation_embedding = relation_embed(constant)

            # rotate the phase pressure

            list_phase_pressure_recomb = []
            num_phase = self.num_actions

            for i in range(num_phase):
                for j in range(num_phase):
                    if i != j:
                        list_phase_pressure_recomb.append(
                            concatenate([list_phase_pressure[i], list_phase_pressure[j]],
                                        name="concat_compete_phase_%d_%d_%d" % (i, j, slice_ind)))

            list_phase_pressure_recomb = concatenate(list_phase_pressure_recomb, name="concat_all_%d" % slice_ind)
            feature_map = Reshape((8, 7, 32))(list_phase_pressure_recomb)
            lane_conv = _lane_conv(feature_map)
            if self.dic_agent_conf["MERGE"] == "multiply":
                relation_conv = _relation_conv(relation_embedding)
                combine_feature = multiply([lane_conv, relation_conv], name="combine_feature_%d" % slice_ind)
            else:
                print("dic_agent_conf[MERGE] has to be multiply")

            hidden_layer = _combine_conv(combine_feature)
            before_merge = _before_merge(hidden_layer)
            q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values_%d" % slice_ind)(Reshape((8, 7))(before_merge))
            if slice_ind == 0:
                feature = q_values
            else:
                feature = concatenate([feature, q_values])
        feature = Reshape((self.num_agents, self.num_actions))(feature)
        feature = self.MLP(feature, MLP_layers)
        return feature

    def iCAP_2(self, In, MLP_layers):
        """
        embed iCAP into colight, process all intersections at once
        """
        phase_input = Lambda(slice_icap, arguments={'name': 'phase'}, output_shape=[self.num_agents, 8],
                             name="inter_phase")(In[0])  # None, 9, 8
        feature_names = self.dic_traffic_env_conf['LIST_STATE_FEATURE']
        the_feature = [a for a in feature_names if 'phase' not in a and 'adjacency' not in a][0]
        if the_feature == 'lane_num_vehicle':
            feature_dim = self.dic_traffic_env_conf['DIC_FEATURE_DIM']["D_" + the_feature.upper()][0] * self.num_lanes
        else:
            feature_dim = self.dic_traffic_env_conf['DIC_FEATURE_DIM']["D_" + the_feature.upper()][0]
        num_vehicle_input = Lambda(slice_icap, arguments={'name': 'num_vehicle'},  # None, 9, 8/12
                                   output_shape=[self.num_agents, feature_dim], name="inter_num_vehicle")(In[0])
        p = Activation('sigmoid')(Embedding(2, 4, input_length=8)(phase_input))
        d = Dense(4, activation="sigmoid", name="num_vec_mapping")

        dic_lane = {}
        for i, m in enumerate(self.dic_traffic_env_conf["list_lane_order"]):
            if feature_dim == 12:  # 12 means the feature contains right lane, which should be discarded
                if i % 2 == 0:
                    j = i // 2 * 3  # select all but right lanes
                else:
                    j = i // 2 * 3 + 1
                tmp_vec = Lambda(slice_tensor_2, arguments={"index": j}, output_shape=[self.num_agents],
                                 name="vec_%d" % j)(num_vehicle_input)
            else:
                tmp_vec = Lambda(slice_tensor_2, arguments={"index": i}, output_shape=[self.num_agents],
                                 name="vec_%d" % i)(num_vehicle_input)
            tmp_vec = d(Reshape((self.num_agents, 1))(tmp_vec))  # None, 9, 4
            tmp_phase = Lambda(slice_tensor_2, arguments={"index": i}, output_shape=[self.num_agents, 4],
                               name="phase_%d" % i)(p)  # None, 9, 4
            dic_lane[m] = concatenate([tmp_vec, tmp_phase], name="lane_%d" % i)  # None, 9, 8

        list_phase_pressure = []
        lane_embedding = Dense(16, activation="relu", name="lane_embedding")

        for phase in self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']]:
            m1, m2 = phase.split("_")
            list_phase_pressure.append(add([lane_embedding(dic_lane[m1]), lane_embedding(dic_lane[m2])], name=phase))

        for i in range(self.num_agents):
            num_vehicle_slice = Lambda(slice_tensor_3, arguments={"index": i}, output_shape=[feature_dim],
                                       name="veh_%d" % i)(num_vehicle_input)  # None 8
            if feature_dim == 12:
                num_vehicle_slice = Lambda(slice_tensor_4, arguments={"index": use_lane_ind},
                                           output_shape=[8],)(num_vehicle_slice)

            if i == 0:
                constant = Lambda(relation, arguments={"dic_traffic_env_conf": self.dic_traffic_env_conf},
                                  name="constant_%d" % i)(num_vehicle_slice)
            else:
                _constant = Lambda(relation, arguments={"dic_traffic_env_conf": self.dic_traffic_env_conf},
                                   name="constant_%d" % i)(num_vehicle_slice)
                constant = concatenate([constant, _constant])

        constant = Reshape((self.num_agents, 8, 7))(constant)
        relation_embedding = Embedding(2, 4, name="relation_embedding")(constant)

        list_phase_pressure_recomb = []
        num_phase = self.num_actions

        for i in range(num_phase):
            for j in range(num_phase):
                if i != j:
                    list_phase_pressure_recomb.append(
                        concatenate([list_phase_pressure[i], list_phase_pressure[j]],
                                    name="concat_compete_phase_%d_%d" % (i, j)))

        list_phase_pressure_recomb = concatenate(list_phase_pressure_recomb, name="concat_all")  # None, 9, 1792
        feature_map = Reshape((self.num_agents, 8, 7, 32))(list_phase_pressure_recomb)  # None, 9, 8, 7, 32
        _lane_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu")
        _relation_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu")
        _hidden_layer = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu")
        _before_merge = Conv2D(1, kernel_size=(1, 1), activation="linear")

        for i in range(self.num_agents):
            feature_map_i = Lambda(slice_tensor_3, arguments={"index": i}, output_shape=[8, 7, 32],
                                   name="feature_map_%d" % i)(feature_map)
            lane_conv = _lane_conv(feature_map_i)
            relation_embedding_i = Lambda(slice_tensor_3, arguments={"index": i}, output_shape=[8, 7, 4],
                                   name="relation_embedding_%d" % i)(relation_embedding)
            relation_conv = _relation_conv(relation_embedding_i)
            combine_feature = multiply([lane_conv, relation_conv])
            hidden_layer = _hidden_layer(combine_feature)
            before_merge = _before_merge(hidden_layer)
            if i == 0:
                q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values_%d" % i)(Reshape((8, 7))(before_merge))
            else:
                _q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values_%d" % i)(Reshape((8, 7))(before_merge))
                q_values = concatenate([q_values, _q_values])
        q_values = Reshape((self.num_agents, self.num_actions))(q_values)
        feature = self.MLP(q_values, MLP_layers)

        return feature

    def MultiHeadsAttModel(self, In_agent, In_neighbor, l=5, d=128, dv=16, dout=128, nv=8, suffix=-1):
        """
        input:[bacth,agent,128]
        output:
        -hidden state: [batch,agent,32]
        -attention: [batch,agent,neighbor]
        """
        """
        agent repr
        """
        print("In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv", In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv)
        # [batch,agent,dim]->[batch,agent,1,dim]
        agent_repr=Reshape((self.num_agents,1,d))(In_agent)

        """
        neighbor repr
        """
        # [batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr=RepeatVector3D(self.num_agents)(In_agent)
        print("neighbor_repr.shape", neighbor_repr.shape)
        # [batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        neighbor_repr=Lambda(lambda x: K.batch_dot(x[0],x[1]))([In_neighbor,neighbor_repr])
        print("neighbor_repr.shape", neighbor_repr.shape)
        """
        attention computation
        """
        # multi-head
        # [batch,agent,1,dim]->[batch,agent,1,dv*nv]
        agent_repr_head=Dense(
            dv*nv,
            activation='relu',
            kernel_initializer='random_normal',
            name='agent_repr_%d' % suffix)(agent_repr)
        # [batch,agent,1,dv,nv]->[batch,agent,nv,1,dv]
        agent_repr_head=Reshape((self.num_agents,1,dv,nv))(agent_repr_head)
        agent_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(agent_repr_head)
        # agent_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,1,dv,nv)),(0,1,4,2,3)))(agent_repr_head)
        # [batch,agent,neighbor,dim]->[batch,agent,neighbor,dv*nv]

        neighbor_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_repr_%d'%suffix)(neighbor_repr)
        # [batch,agent,neighbor,dv,nv]->[batch,agent,nv,neighbor,dv]
        print("DEBUG", neighbor_repr_head.shape)
        print("self.num_agents,self.num_neighbors,dv,nv", self.num_agents,self.num_neighbors, dv, nv)
        neighbor_repr_head=Reshape((self.num_agents,self.num_neighbors, dv, nv))(neighbor_repr_head)
        neighbor_repr_head=Lambda(lambda x: K.permute_dimensions(x,(0, 1, 4, 2, 3)))(neighbor_repr_head)
        # neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,self.num_neighbors,dv,nv)),(0,1,4,2,3)))(neighbor_repr_head)
        # [batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
        att=Lambda(lambda x:K.softmax(K.batch_dot(x[0],x[1],axes=[4,4])))([agent_repr_head,neighbor_repr_head])
        # [batch,agent,nv,1,neighbor]->[batch,agent,nv,neighbor]
        att_record=Reshape((self.num_agents,nv,self.num_neighbors))(att)

        # self embedding again
        neighbor_hidden_repr_head = Dense(
            dv*nv,
            activation='relu',
            kernel_initializer='random_normal',
            name='neighbor_hidden_repr_%d' % suffix)(neighbor_repr)
        neighbor_hidden_repr_head = Reshape((self.num_agents,self.num_neighbors, dv, nv))(neighbor_hidden_repr_head)
        neighbor_hidden_repr_head = Lambda(lambda x:K.permute_dimensions(x, (0, 1, 4, 2, 3)))(neighbor_hidden_repr_head)
        out=Lambda(lambda x: K.mean(K.batch_dot(x[0], x[1]), axis=2))([att,neighbor_hidden_repr_head])
        out=Reshape((self.num_agents,dv))(out)
        out = Dense(dout, activation="relu",kernel_initializer='random_normal',name='MLP_after_relation_%d'%suffix)(out)
        return out,att_record

    def adjacency_index2matrix(self,adjacency_index):
        # adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and 
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """ 
        # [batch,agents,neighbors]
        adjacency_index_new=np.sort(adjacency_index, axis=-1)
        l = to_categorical(adjacency_index_new, num_classes=self.num_agents)
        return l

    def action_att_predict(self,state,total_features=[],total_adjs=[],bar=False):
        # state:[batch,agent,features and adj]
        # return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
        batch_size = len(state)
        if total_features==[] and total_adjs==[]:
            total_features,total_adjs=list(),list()
            for i in range(batch_size): 
                feature=[]
                adj=[] 
                for j in range(self.num_agents):
                    observation=[]
                    for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                        if 'adjacency' in feature_name:
                            continue
                        if feature_name == "cur_phase":
                            if len(state[i][j][feature_name])==1:
                                # choose_action
                                observation.extend(self.dic_traffic_env_conf['phase_expansion'][state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name == "lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                        elif feature_name == "pressure_of_movement":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)
                    adj.append(state[i][j]['adjacency_matrix'])
                total_features.append(feature)
                total_adjs.append(adj)
            # feature:[agents,feature]
            total_features = np.reshape(np.array(total_features), [batch_size, self.num_agents, -1])
            total_adjs = self.adjacency_index2matrix(np.array(total_adjs))
            # adj:[agent,neighbors]
        if bar:
            all_output = self.q_network_bar.predict([total_features, total_adjs])
        else:
            all_output = self.q_network.predict([total_features, total_adjs])
        action, attention = all_output[0], all_output[1]

        # out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        if len(action)>1:
            return total_features, total_adjs, action, attention

        # [batch,agent,1]
        max_action=np.expand_dims(np.argmax(action,axis=-1),axis=-1)
        random_action=np.reshape(np.random.randint(self.num_actions,size=1*self.num_agents),(1,self.num_agents,1))
        # [batch,agent,2]
        possible_action=np.concatenate([max_action,random_action],axis=-1)
        selection=np.random.choice(
            [0,1],
            size=batch_size*self.num_agents,
            p=[1-self.dic_agent_conf["EPSILON"],self.dic_agent_conf["EPSILON"]])
        act=possible_action.reshape((batch_size*self.num_agents,2))[np.arange(batch_size*self.num_agents),selection]
        act=np.reshape(act,(batch_size,self.num_agents))
        return act,attention

    def choose_action(self, count, state):

        """
        choose the best action for current state 
        -input: state: [batch,agent,feature]  adj: [batch,agent,neighbors,agents]
        -output: out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        """
        act, attention = self.action_att_predict([state])
        return act[0], attention[0]

    def prepare_Xs_Y(self, memory, dic_exp_conf):

        if dic_exp_conf["USE_FRAP"]:
            feature_names = self.dic_traffic_env_conf['LIST_STATE_FEATURE']
            the_feature = [a for a in feature_names if 'phase' not in a and 'adjacency' not in a][0]
            ind_end = len(memory)
            print("memory size before forget: {0}".format(ind_end))

            if self.dic_agent_conf['PRIORITY']:  # TODO: what is this? if needed, then extend to multi-intersection
                print("priority")
                sample_slice = []
                num_sample_list = [
                    int(self.dic_agent_conf["MAX_MEMORY_LEN"] * 1 / 4),
                    int(self.dic_agent_conf["MAX_MEMORY_LEN"] * 1 / 4),
                    int(self.dic_agent_conf["MAX_MEMORY_LEN"] * 1 / 4),
                    int(self.dic_agent_conf["MAX_MEMORY_LEN"] * 1 / 4),
                ]
                for i in range(ind_end - 1, -1, -1):
                    one_sample = memory[i]
                    # print("one_sample", one_sample)
                    # ave_num_veh = int(np.average(np.array(one_sample[0]['lane_num_vehicle'])))
                    ave_num_veh = max(np.array(one_sample[0][0][the_feature]))
                    interval_id = ave_num_veh // 10
                    if num_sample_list[interval_id] > 0:
                        sample_slice.append(one_sample)
                        num_sample_list[interval_id] -= 1

                    if np.sum(np.array(num_sample_list)) == 0:
                        break
                print("end priority", num_sample_list)
                sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(sample_slice))
                sample_slice = random.sample(sample_slice, sample_size)

            else:
                ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
                memory_after_forget = memory[ind_sta: ind_end]
                print("memory size after forget:", len(memory_after_forget))

                # sample the memory
                sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
                sample_slice = random.sample(memory_after_forget, sample_size)
                print("memory samples number:", sample_size)

            if self.dic_agent_conf["ROTATION_INPUT"]:

                new_sample_slice = []
                for sample in sample_slice:
                    # print(sample[0]['lane_num_vehicle'], sample[0]['cur_phase'])
                    rotation_matrix = rotation_matrix_4_p
                    new_sample = copy.deepcopy(sample)
                    for j in range(self.num_agents):
                        if the_feature == 'lane_num_vehicle':
                            feature = np.array(new_sample[j][0][the_feature])
                            feature[use_lane_ind] = np.dot(feature[use_lane_ind], rotation_matrix)
                            new_sample[j][0][the_feature] = list(feature)
                            new_sample[j][0]['cur_phase'] = np.dot(new_sample[j][0]['cur_phase'], rotation_matrix)

                            feature = np.array(new_sample[j][2][the_feature])
                            feature[use_lane_ind] = np.dot(feature[use_lane_ind], rotation_matrix)
                            new_sample[j][2][the_feature] = list(feature)
                            new_sample[j][2]['cur_phase'] = np.dot(new_sample[j][2]['cur_phase'], rotation_matrix)
                        else:
                            new_sample[j][0][the_feature] = np.dot(new_sample[j][0][the_feature], rotation_matrix)
                            new_sample[j][0]['cur_phase'] = np.dot(new_sample[j][0]['cur_phase'], rotation_matrix)
                            new_sample[j][2][the_feature] = np.dot(new_sample[j][2][the_feature], rotation_matrix)
                            new_sample[j][2]['cur_phase'] = np.dot(new_sample[j][2]['cur_phase'], rotation_matrix)
                    new_sample_slice.append(sample)
                    new_sample_slice.append(new_sample)
                sample_slice = new_sample_slice
        else:
            ind_end = len(memory)
            print("memory size before forget: {0}".format(ind_end))
            # use all the samples to pretrain, i.e., without forgetting
            if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
                sample_slice = memory
            # forget
            else:
                ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
                memory_after_forget = memory[ind_sta: ind_end]
                print("memory size after forget:", len(memory_after_forget))

                # sample the memory
                sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
                sample_slice = random.sample(memory_after_forget, sample_size)
                print("memory samples number:", sample_size)

        _state = []
        _next_state = []
        _action = []
        _reward = []

        for i in range(len(sample_slice)):  
            _state.append([])
            _next_state.append([])
            _action.append([])
            _reward.append([])
            for j in range(self.num_agents):
                state, action, next_state, reward, _ = sample_slice[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
                _action[i].append(action)
                _reward[i].append(reward)

        # target: [#agents,#samples,#num_actions]
        _features, _adjs, q_values, _ = self.action_att_predict(_state)
        _next_features, _next_adjs, _, attention = self.action_att_predict(_next_state)
        # target_q_values:[batch,agent,action]
        _, _, target_q_values, _ = self.action_att_predict(
            _next_state,
            total_features=_next_features,
            total_adjs=_next_adjs,
            bar=True)

        for i in range(len(sample_slice)):
            for j in range(self.num_agents):
                q_values[i][j][_action[i][j]] = _reward[i][j] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(target_q_values[i][j])

        # self.Xs should be: [#agents,#samples,#features+#]
        self.Xs = [_features,_adjs]
        self.Y = q_values.copy()
        self.Y_total = [q_values.copy()]
        self.Y_total.append(attention)
        return 

    # TODO: MLP_layers should be defined in the conf file
    # TODO: CNN_layers should be defined in the conf file
    # TODO: CNN_heads should be defined in the conf file
    # TODO: Output_layers should be degined in the conf file
    def build_network(
        self,
        MLP_layers = [32,32],
        # CNN_layers=[[32,32]],#[[4,32],[4,32]],
        # CNN_heads=[1],#[8,8],
        Output_layers = []):
        CNN_layers = self.CNN_layers
        CNN_heads = [1]*len(CNN_layers)
        """
        layer definition
        """
        start_time=time.time()
        assert len(CNN_layers)==len(CNN_heads)

        In = list()
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        # In: [batch,agent,feature]
        # In: [batch,agent,neighbors,agents]
        In.append(Input(shape=[self.num_agents, self.len_feature], name="feature"))
        In.append(Input(shape=(self.num_agents, self.num_neighbors, self.num_agents), name="adjacency_matrix"))
        # In[0] has to be form of {'cur_phase': Input(8), 'lane_num_vehicle': Input(8)}

        Input_end_time = time.time()
        """
        Currently, the MLP layer 
        -input: [batch,agent,feature_dim]
        -outpout: [#agent,batch,128]
        """

        # =================== iCAP part ======================
        if self.dic_agent_conf["USE_FRAP"]:
            feature = self.iCAP_2(In, MLP_layers)
        else:
            feature = self.MLP(In[0], MLP_layers)
        # =================== end of iCAP part ======================

        Embedding_end_time = time.time()

        # TODO: remove the dense setting
        # feature:[batch,agents,feature_dim]
        att_record_all_layers=list()
        print("CNN_heads:", CNN_heads)
        for CNN_layer_index,CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:", CNN_heads[CNN_layer_index])
            if CNN_layer_index==0:
                h, att_record=self.MultiHeadsAttModel(
                    feature,
                    In[1],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            else:
                h, att_record=self.MultiHeadsAttModel(
                    h,
                    In[1],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            att_record_all_layers.append(att_record)

        if len(CNN_layers) > 1:
            att_record_all_layers = Concatenate(axis=1)(att_record_all_layers)
        else:
            att_record_all_layers = att_record_all_layers[0]

        att_record_all_layers=Reshape(
            (len(CNN_layers),self.num_agents,CNN_heads[-1],self.num_neighbors)
            )(att_record_all_layers)

        # TODO: remove dense net
        for layer_index,layer_size in enumerate(Output_layers):
                h = Dense(layer_size,
                          activation='relu',
                          kernel_initializer='random_normal',
                          name='Dense_q_%d' % layer_index)(h)
        # action prediction layer
        # [batch,agent,32]->[batch,agent,action]
        out = Dense(self.num_actions, kernel_initializer='random_normal',name='action_layer')(h)
        # out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        model = Model(inputs=In,outputs=[out, att_record_all_layers])

        if self.att_regulatization:
            model.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"],'kullback_leibler_divergence'],
                loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
        else:
            model.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1,0])
        # model.compile(optimizer=Adam(lr = 0.0001), loss='mse')
        model.summary()
        network_end=time.time()
        print('build_Input_end_time：', Input_end_time-start_time)
        print('embedding_time:', Embedding_end_time-Input_end_time)
        print('total time:', network_end-start_time)
        return model

    def build_memory(self):

        return []

    def train_network(self, dic_exp_conf):

        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        # hist = self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs,
        #                           shuffle=False,
        #                           verbose=2, validation_split=0.3, callbacks=[early_stopping])
        hist = self.q_network.fit(self.Xs, self.Y_total, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=2, validation_split=0.3,
                                  callbacks=[early_stopping, TensorBoard(log_dir='./temp.tensorboard')])

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''
        network_weights = network_copy.get_weights()
        network = self.build_network()
        network.set_weights(network_weights)

        if self.att_regulatization:
            network.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"] for i in range(self.num_agents)]+['kullback_leibler_divergence'],
                loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
        else:
            network.compile(
                optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1,0])

        return network

    def load_network(self, file_name):
        self.q_network = self.build_network()
        self.q_network.load_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "{}.h5".format(file_name)))
        self.q_network.compile(
            optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
            loss=self.dic_agent_conf["LOSS_FUNCTION"],
            loss_weights=[1, 0])
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name):
        self.q_network_bar = self.build_network()
        self.q_network_bar.load_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "{}.h5".format(file_name)))
        self.q_network_bar.compile(
            optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
            loss=self.dic_agent_conf["LOSS_FUNCTION"],
            loss_weights=[1, 0])

        print("succeed in loading model bar %s" % file_name)

    def save_network(self, file_name):  # no need to save model structure since it will be built
        self.q_network.save_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "{}.h5".format(file_name)))

    def save_network_bar(self, file_name):
        self.q_network_bar.save_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "{}_bar.h5".format(file_name)))


if __name__ == '__main__':
    dic_agent_conf={
        'att_regularization': False, 
        'rularization_rate': 0.03, 
        'LEARNING_RATE': 0.001, 
        'SAMPLE_SIZE': 1000, 
        'BATCH_SIZE': 20, 
        'EPOCHS': 100, 
        'UPDATE_Q_BAR_FREQ': 5, 
        'UPDATE_Q_BAR_EVERY_C_ROUND': False, 
        'GAMMA': 0.8, 
        'MAX_MEMORY_LEN': 10000, 
        'PATIENCE': 10, 
        'D_DENSE': 20, 
        'N_LAYER': 2, 
        'EPSILON': 0.8, 
        'EPSILON_DECAY': 0.95, 
        'MIN_EPSILON': 0.2, 
        'LOSS_FUNCTION': 'mean_squared_error', 
        'SEPARATE_MEMORY': False, 
        'NORMAL_FACTOR': 20, 
        'TRAFFIC_FILE': 'sumo_1_3_300_connect_all.xml'}
    dic_traffic_env_conf={
        'ACTION_PATTERN': 'set', 
        'NUM_INTERSECTIONS': 1000, 
        'TOP_K_ADJACENCY': 1000, 
        'MIN_ACTION_TIME': 10, 
        'YELLOW_TIME': 5, 
        'ALL_RED_TIME': 0, 
        'NUM_PHASES': 2, 
        'NUM_LANES': 1, 
        'ACTION_DIM': 2, 
        'MEASURE_TIME': 10, 
        'IF_GUI': False, 
        'DEBUG': False, 
        'INTERVAL': 1, 
        'THREADNUM': 8, 
        'SAVEREPLAY': True, 
        'RLTRAFFICLIGHT': True, 
        'DIC_FEATURE_DIM': {'D_LANE_QUEUE_LENGTH': (4,), 'D_LANE_NUM_VEHICLE': (4,), 'D_COMING_VEHICLE': (4,), 'D_LEAVING_VEHICLE': (4,), 'D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1': (4,), 'D_CUR_PHASE': (8,), 'D_NEXT_PHASE': (8,), 'D_TIME_THIS_PHASE': (1,), 'D_TERMINAL': (1,), 'D_LANE_SUM_WAITING_TIME': (4,), 'D_VEHICLE_POSITION_IMG': (4, 60), 'D_VEHICLE_SPEED_IMG': (4, 60), 'D_VEHICLE_WAITING_TIME_IMG': (4, 60), 'D_PRESSURE': (1,), 'D_ADJACENCY_MATRIX': (3,)}, 
        'LIST_STATE_FEATURE': ['cur_phase', 'lane_num_vehicle', 'adjacency_matrix'], 
        'DIC_REWARD_INFO': {'flickering': 0, 'sum_lane_queue_length': 0, 'sum_lane_wait_time': 0, 'sum_lane_num_vehicle_left': 0, 'sum_duration_vehicle_left': 0, 'sum_num_vehicle_been_stopped_thres01': 0, 'sum_num_vehicle_been_stopped_thres1': 0, 'pressure': -0.25}, 
        'LANE_NUM': {'LEFT': 1, 'RIGHT': 1, 'STRAIGHT': 1}, 
        'PHASE': {'sumo': {0: [0, 1, 0, 1, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 1, 0, 1]}, 'anon': {1: [0, 1, 0, 1, 0, 0, 0, 0], 2: [0, 0,0, 0, 0, 1, 0, 1], 3: [1, 0, 1, 0, 0, 0, 0, 0], 4: [0, 0, 0, 0, 1, 0, 1, 0]}}, 
        'ONE_MODEL': False, 
        'NUM_AGENTS': 1, 
        'SIMULATOR_TYPE': 'sumo', 
        'BINARY_PHASE_EXPANSION': True, 
        'NUM_ROW': 3, 
        'NUM_COL': 1, 
        'TRAFFIC_FILE': 'sumo_1_3_300_connect_all.xml', 
        'ROADNET_FILE': 'roadnet_1_3.json'}
    dic_path={
        'PATH_TO_MODEL': 'model/0106_afternoon_1x3_300_GCN_time_test/sumo_1_3_300_connect_all.xml_01_06_17_23_51', 
        'PATH_TO_WORK_DIRECTORY': 'records/0106_afternoon_1x3_300_GCN_time_test/sumo_1_3_300_connect_all.xml_01_06_17_23_51', 
        'PATH_TO_DATA': 'data/template_lsr/1_3', 
        'PATH_TO_PRETRAIN_MODEL': 'model/initial/sumo_1_3_300_connect_all.xml', 
        'PATH_TO_PRETRAIN_WORK_DIRECTORY':'records/initial/sumo_1_3_300_connect_all.xml', 
        'PATH_TO_PRETRAIN_DATA': 'data/template', 
        'PATH_TO_AGGREGATE_SAMPLES': 'records/initial', 
        'PATH_TO_ERROR': 'errors/0106_afternoon_1x3_300_GCN_time_test'}
    cnt_round=200
    one_agent=CoLightAgent(
        dic_agent_conf=dic_agent_conf, 
        dic_traffic_env_conf=dic_traffic_env_conf, 
        dic_path=dic_path, 
        cnt_round=cnt_round,        
    )
    one_model=one_agent.build_network()
    one_agent.build_network_from_copy(one_model)




import numpy as np 
import os 
import pickle  
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
from keras.layers.core import Activation
from keras.utils import np_utils,to_categorical
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping, TensorBoard

# SEED=6666
# random.seed(SEED)
# np.random.seed(SEED)
# tf.set_random_seed(SEED)


class RepeatVector3D(Layer):
    def __init__(self,times,**kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.times, input_shape[1],input_shape[2])

    def call(self, inputs):
        #[batch,agent,dim]->[batch,1,agent,dim]
        #[batch,1,agent,dim]->[batch,agent,agent,dim]

        return K.tile(K.expand_dims(inputs,1),[1,self.times,1,1])


    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CoLightAgent(Agent): 
    def __init__(self, 
        dic_agent_conf=None, 
        dic_traffic_env_conf=None, 
        dic_path=None, 
        cnt_round=None, 
        best_round=None, bar_round=None,intersection_id="0"):
        """
        #1. compute the (dynamic) static Adjacency matrix, compute for each state
        -2. #neighbors: 5 (1 itself + W,E,S,N directions)
        -3. compute len_features
        -4. self.num_actions
        """
        super(CoLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,intersection_id)

        self.att_regulatization=dic_agent_conf['att_regularization']
        self.CNN_layers=dic_agent_conf['CNN_layers']
        
        #TODO: n_agents should pass as parameter
        self.num_agents=dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors=min(dic_traffic_env_conf['TOP_K_ADJACENCY'],self.num_agents)
        self.vec=np.zeros((1,self.num_neighbors))
        self.vec[0][0]=1

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.len_feature=self.compute_len_feature()
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
                    #TODO how to load network for multiple intersections?
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
            #the 0-th model is pretrained model
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
        return sum(len_feature)

    """
    components of the network
    1. MLP encoder of features
    2. CNN layers
    3. q network
    """
    def MLP(self,In_0,layers=[128,128]):
        """
        Currently, the MLP layer 
        -input: [batch,#agents,feature_dim]
        -outpout: [batch,#agents,128]
        """
        # In_0 = Input(shape=[self.num_agents,self.len_feature])
        for layer_index,layer_size in enumerate(layers):
            if layer_index==0:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(In_0)
            else:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(h)

        return h





    def MultiHeadsAttModel(self,In_agent,In_neighbor,l=5, d=128, dv=16, dout=128, nv = 8,suffix=-1):
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
        #[batch,agent,dim]->[batch,agent,1,dim]
        agent_repr=Reshape((self.num_agents,1,d))(In_agent)

        """
        neighbor repr
        """
        #[batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr=RepeatVector3D(self.num_agents)(In_agent)
        print("neighbor_repr.shape", neighbor_repr.shape)
        #[batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        neighbor_repr=Lambda(lambda x:K.batch_dot(x[0],x[1]))([In_neighbor,neighbor_repr])
        print("neighbor_repr.shape", neighbor_repr.shape)
        """
        attention computation
        """
        #multi-head
        #[batch,agent,1,dim]->[batch,agent,1,dv*nv]
        agent_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='agent_repr_%d'%suffix)(agent_repr)
        #[batch,agent,1,dv,nv]->[batch,agent,nv,1,dv]
        agent_repr_head=Reshape((self.num_agents,1,dv,nv))(agent_repr_head)
        agent_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(agent_repr_head)
        #agent_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,1,dv,nv)),(0,1,4,2,3)))(agent_repr_head)
        #[batch,agent,neighbor,dim]->[batch,agent,neighbor,dv*nv]

        neighbor_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_repr_%d'%suffix)(neighbor_repr)
        #[batch,agent,neighbor,dv,nv]->[batch,agent,nv,neighbor,dv]
        print("DEBUG",neighbor_repr_head.shape)
        print("self.num_agents,self.num_neighbors,dv,nv", self.num_agents,self.num_neighbors,dv,nv)
        neighbor_repr_head=Reshape((self.num_agents,self.num_neighbors,dv,nv))(neighbor_repr_head)
        neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_repr_head)
        #neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(K.reshape(x,(-1,self.num_agents,self.num_neighbors,dv,nv)),(0,1,4,2,3)))(neighbor_repr_head)        
        #[batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
        att=Lambda(lambda x:K.softmax(K.batch_dot(x[0],x[1],axes=[4,4])))([agent_repr_head,neighbor_repr_head])
        #[batch,agent,nv,1,neighbor]->[batch,agent,nv,neighbor]
        att_record=Reshape((self.num_agents,nv,self.num_neighbors))(att)


        #self embedding again
        neighbor_hidden_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_hidden_repr_%d'%suffix)(neighbor_repr)
        neighbor_hidden_repr_head=Reshape((self.num_agents,self.num_neighbors,dv,nv))(neighbor_hidden_repr_head)
        neighbor_hidden_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_hidden_repr_head)
        out=Lambda(lambda x:K.mean(K.batch_dot(x[0],x[1]),axis=2))([att,neighbor_hidden_repr_head])
        out=Reshape((self.num_agents,dv))(out)
        out = Dense(dout, activation = "relu",kernel_initializer='random_normal',name='MLP_after_relation_%d'%suffix)(out)
        return out,att_record





    def adjacency_index2matrix(self,adjacency_index):
        #adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and 
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """ 
        #[batch,agents,neighbors]
        adjacency_index_new=np.sort(adjacency_index,axis=-1)
        l = to_categorical(adjacency_index_new,num_classes=self.num_agents)
        return l

    def action_att_predict(self,state,total_features=[],total_adjs=[],bar=False):
        #state:[batch,agent,features and adj]
        #return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
        batch_size=len(state)
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
                                #choose_action
                                observation.extend(self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                                            [state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name=="lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)
                    adj.append(state[i][j]['adjacency_matrix'])
                total_features.append(feature)
                total_adjs.append(adj)
            #feature:[agents,feature]
            total_features=np.reshape(np.array(total_features),[batch_size,self.num_agents,-1])
            total_adjs=self.adjacency_index2matrix(np.array(total_adjs))
            #adj:[agent,neighbors]   
        if bar:
            all_output= self.q_network_bar.predict([total_features,total_adjs])
        else:
            all_output= self.q_network.predict([total_features,total_adjs])
        action,attention =all_output[0],all_output[1]

        #out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        if len(action)>1:
            return total_features,total_adjs,action,attention

        #[batch,agent,1]
        max_action=np.expand_dims(np.argmax(action,axis=-1),axis=-1)
        random_action=np.reshape(np.random.randint(self.num_actions,size=1*self.num_agents),(1,self.num_agents,1))
        #[batch,agent,2]
        possible_action=np.concatenate([max_action,random_action],axis=-1)
        selection=np.random.choice(
            [0,1],
            size=batch_size*self.num_agents,
            p=[1-self.dic_agent_conf["EPSILON"],self.dic_agent_conf["EPSILON"]])
        act=possible_action.reshape((batch_size*self.num_agents,2))[np.arange(batch_size*self.num_agents),selection]
        act=np.reshape(act,(batch_size,self.num_agents))
        return act,attention


    def choose_action(self, count, state):

        ''' 
        choose the best action for current state 
        -input: state: [batch,agent,feature]  adj: [batch,agent,neighbors,agents]
        -output: out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        '''
        act,attention=self.action_att_predict([state])
        return act[0],attention[0] 


    def prepare_Xs_Y(self, memory, dic_exp_conf):
        """
        
        """
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
        _action=[]
        _reward=[]

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


        #target: [#agents,#samples,#num_actions]    
        _features,_adjs,q_values,_=self.action_att_predict(_state)   
        _next_features,_next_adjs,_,attention= self.action_att_predict(_next_state)
        #target_q_values:[batch,agent,action]
        _,_,target_q_values,_= self.action_att_predict(
            _next_state,
            total_features=_next_features,
            total_adjs=_next_adjs,
            bar=True)

        for i in range(len(sample_slice)):
            for j in range(self.num_agents):
                q_values[i][j][_action[i][j]] = _reward[i][j] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(target_q_values[i][j])


        #self.Xs should be: [#agents,#samples,#features+#]
        self.Xs = [_features,_adjs]
        self.Y=q_values.copy()
        self.Y_total = [q_values.copy()]
        self.Y_total.append(attention)
        return 

    #TODO: MLP_layers should be defined in the conf file
    #TODO: CNN_layers should be defined in the conf file
    #TODO: CNN_heads should be defined in the conf file
    #TODO: Output_layers should be degined in the conf file
    def build_network(
        self,
        MLP_layers=[32,32], 
        # CNN_layers=[[32,32]],#[[4,32],[4,32]],
        # CNN_heads=[1],#[8,8],
        Output_layers=[]):
        CNN_layers=self.CNN_layers 
        CNN_heads=[1]*len(CNN_layers)
        """
        layer definition
        """
        start_time=time.time()
        assert len(CNN_layers)==len(CNN_heads)

        In=list()
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        #In: [batch,agent,feature]
        #In: [batch,agent,neighbors,agents]
        In.append(Input(shape=[self.num_agents,self.len_feature],name="feature"))
        In.append(Input(shape=(self.num_agents,self.num_neighbors,self.num_agents),name="adjacency_matrix"))


        Input_end_time=time.time()
        """
        Currently, the MLP layer 
        -input: [batch,agent,feature_dim]
        -outpout: [#agent,batch,128]
        """
        feature=self.MLP(In[0],MLP_layers)

        Embedding_end_time=time.time()


        #TODO: remove the dense setting
        #feature:[batch,agents,feature_dim]
        att_record_all_layers=list()
        print("CNN_heads:", CNN_heads)
        for CNN_layer_index,CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:",CNN_heads[CNN_layer_index])
            if CNN_layer_index==0:
                h,att_record=self.MultiHeadsAttModel(
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
                h,att_record=self.MultiHeadsAttModel(
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

        if len(CNN_layers)>1:
            att_record_all_layers=Concatenate(axis=1)(att_record_all_layers)
        else:
            att_record_all_layers=att_record_all_layers[0]

        att_record_all_layers=Reshape(
            (len(CNN_layers),self.num_agents,CNN_heads[-1],self.num_neighbors)
            )(att_record_all_layers)

        
        #TODO remove dense net
        for layer_index,layer_size in enumerate(Output_layers):
                h=Dense(layer_size,activation='relu',kernel_initializer='random_normal',name='Dense_q_%d'%layer_index)(h)
        #action prediction layer
        #[batch,agent,32]->[batch,agent,action]
        out = Dense(self.num_actions,kernel_initializer='random_normal',name='action_layer')(h)
        #out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        model=Model(inputs=In,outputs=[out,att_record_all_layers])

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
        print('build_Input_end_time：',Input_end_time-start_time)
        print('embedding_time:',Embedding_end_time-Input_end_time)
        print('total time:',network_end-start_time)
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
                                  callbacks=[early_stopping,TensorBoard(log_dir='./temp.tensorboard')])

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"RepeatVector3D": RepeatVector3D})
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

    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        self.q_network = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D':RepeatVector3D})
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(
            os.path.join(file_path, "%s.h5" % file_name),
            custom_objects={'RepeatVector3D':RepeatVector3D})
        print("succeed in loading model %s"%file_name) 

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))



if __name__=='__main__':
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




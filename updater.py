import pickle
import os
from config import DIC_AGENTS,DIC_ENVS
import pandas as pd
import shutil
import pandas as pd
import time
from multiprocessing import Pool
import traceback
import random
import numpy as np


class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, best_round=None, bar_round=None):

        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_exp_conf = dic_exp_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None


        #temporay path_to_log
        self.path_to_log=os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_0", "generator_0")
        env_tmp = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                              path_to_log = self.path_to_log,
                              path_to_work_directory = self.dic_path["PATH_TO_WORK_DIRECTORY"],
                              dic_traffic_env_conf = self.dic_traffic_env_conf)        
        env_tmp.reset()

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_exp_conf["MODEL_NAME"]
            if agent_name=='CoLight_Signal':
                agent= DIC_AGENTS[agent_name](
                    self.dic_agent_conf, self.dic_traffic_env_conf,
                    self.dic_path, self.cnt_round, 
                    inter_info=env_tmp.list_intersection,
                    intersection_id=str(i))
            else:
                agent= DIC_AGENTS[agent_name](
                    self.dic_agent_conf, self.dic_traffic_env_conf,
                    self.dic_path, self.cnt_round, intersection_id=str(i))
            self.agents.append(agent)

    def load_sample(self, i):
        sample_set = []
        try:
            if self.dic_exp_conf["PRETRAIN"]:
                    sample_file = open(os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"],
                                                "train_round", "total_samples" + ".pkl"), "rb")
            elif self.dic_exp_conf["AGGREGATE"]:
                sample_file = open(os.path.join(self.dic_path["PATH_TO_AGGREGATE_SAMPLES"],
                                                "aggregate_samples.pkl"), "rb")
            else:
                sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                                "total_samples_inter_{0}".format(i) + ".pkl"), "rb")
            try:
                while True:
                    sample_set += pickle.load(sample_file)
            except EOFError:
                sample_file.close()
                pass
        except Exception as e:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a")
            f.write("Fail to load samples for inter {0}\n".format(i))
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        if i %100 ==0:
            print("load_sample for inter {0}".format(i))
        return sample_set

    def load_hidden_states_with_forget(self): # hidden state is a list [#time, agent, # dim]
        hidden_states_set = []
        try:
            hidden_state_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                                "total_hidden_states.pkl"), "rb")
            try:
                while True:
                    hidden_states_set.append(pickle.load(hidden_state_file))
                    hidden_states_set = np.vstack(hidden_states_set)
                    ind_end = len(hidden_states_set)
                    print("hidden_state_set shape: ",hidden_states_set.shape)
                    if self.dic_exp_conf["PRETRAIN"] or self.dic_exp_conf["AGGREGATE"]:
                        pass
                    else:
                        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
                        hidden_states_after_forget = hidden_states_set[ind_sta: ind_end]
                        hidden_states_set = [np.array([hidden_states_after_forget[k] for k in self.sample_indexes])]
            except EOFError:
                hidden_state_file.close()
                pass
        except Exception as e:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info.txt"), "a")
            f.write("Fail to load hidden_states for inter\n")
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        return hidden_states_set

    def load_hidden_states(self): # hidden state is a list [#time, agent, # dim]
        hidden_states_set = []
        try:
            hidden_state_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                                "total_hidden_states.pkl"), "rb")
            try:
                while True:
                    hidden_states_set.append(pickle.load(hidden_state_file))
            except EOFError:
                pass
        except Exception as e:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info.txt"), "a")
            f.write("Fail to load hidden_states for inter\n")
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        return hidden_states_set


    def load_sample_with_forget(self, i):
        '''
        Load sample for each intersection, with forget
        :param i:
        :return: a list of samples with fixed indexes
        '''

        sample_set = []
        try:
            if self.dic_exp_conf["PRETRAIN"]:
                    sample_file = open(os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"],
                                                "train_round", "total_samples" + ".pkl"), "rb")
            elif self.dic_exp_conf["AGGREGATE"]:
                sample_file = open(os.path.join(self.dic_path["PATH_TO_AGGREGATE_SAMPLES"],
                                                "aggregate_samples.pkl"), "rb")
            else:
                sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                                "total_samples_inter_{0}".format(i) + ".pkl"), "rb")
            try:
                while True:
                    cur_round_sample_set = pickle.load(sample_file)
                    ind_end = len(cur_round_sample_set)
                    if self.dic_exp_conf["PRETRAIN"] or self.dic_exp_conf["AGGREGATE"]:
                        pass
                    else:
                        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
                        memory_after_forget = cur_round_sample_set[ind_sta: ind_end]
                        # print("memory size after forget:", len(memory_after_forget))

                        # sample the memory
                        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
                        if self.sample_indexes is None:
                            self.sample_indexes = random.sample(range(len(memory_after_forget)), sample_size)
                        sample_set = [memory_after_forget[k] for k in self.sample_indexes]
                        # print("memory samples number:", sample_size)
                        # print(self.sample_indexes)
                    sample_set += cur_round_sample_set
            except EOFError:
                pass
        except Exception as e:
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a")
            f.write("Fail to load samples for inter {0}\n".format(i))
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            pass
        if i %100 == 0:
            print("load_sample for inter {0}".format(i))
        return sample_set


    def load_sample_for_agents(self):
        # TODO should be number of agents
        start_time = time.time()
        print("Start load samples at", start_time)
        if self.dic_exp_conf['MODEL_NAME'] not in ["GCN","CoLight"]:
            if self.dic_traffic_env_conf["ONE_MODEL"] or self.dic_exp_conf['MODEL_NAME'] in ["SimpleDQNOne"]: # for one model
                sample_set_all = []
                for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                    sample_set = self.load_sample_with_forget(i)
                    sample_set_all.extend(sample_set)

                self.agents[0].prepare_Xs_Y(sample_set_all, self.dic_exp_conf)

            else:
                for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                    sample_set = self.load_sample(i)
                    self.agents[i].prepare_Xs_Y(sample_set, self.dic_exp_conf)
        else:
            samples_gcn_df = None
            if False: # Todo decide multi-process
                for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                    sample_set = self.load_sample(i)
                    if len(sample_set) == 0:
                        continue
                    samples_set_df = pd.DataFrame.from_records(sample_set,columns= ['state','action','next_state','inst_reward','reward','time','generator'])
                    samples_set_df['input'] = samples_set_df[['state','action','next_state','inst_reward','reward']].values.tolist()
                    samples_set_df.drop(['state','action','next_state','inst_reward','reward'], axis=1, inplace=True)
                    # samples_set_df['inter_id'] = i
                    if samples_gcn_df is None:
                        samples_gcn_df = samples_set_df
                    else:
                        # print(samples_set_df[['time','generator']])
                        samples_gcn_df = pd.merge(samples_gcn_df, samples_set_df, how='inner',
                                                  on=["generator",'time'], suffixes=('','_{0}'.format(i)))
                intersection_input_columns = ['input'] + ['input_{0}'.format(i+1) for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']-1)]
                for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
                    sample_set_list = samples_gcn_df[intersection_input_columns].values.tolist()
                    self.agents[i].prepare_Xs_Y(sample_set_list, self.dic_exp_conf)
            elif False :# True:
                samples_gcn_df = []
                print("start get samples")
                get_samples_start_time = time.time()
                for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                    sample_set = self.load_sample(i)
                    samples_set_df = pd.DataFrame.from_records(sample_set,columns= ['state','action','next_state','inst_reward','reward','time','generator'])
                    samples_set_df['input'] = samples_set_df[['state','action','next_state','inst_reward','reward']].values.tolist()
                    samples_set_df.drop(['state','action','next_state','inst_reward','reward','time','generator'], axis=1, inplace=True)
                    # samples_set_df['inter_id'] = i
                    samples_gcn_df.append(samples_set_df['input'])
                    if i%100 == 0:
                        print("inter {0} samples_set_df.shape: ".format(i), samples_set_df.shape)
                samples_gcn_df = pd.concat(samples_gcn_df, axis=1)
                print("samples_gcn_df.shape :", samples_gcn_df.shape)
                print("Getting samples time: ", time.time()-get_samples_start_time)

                for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
                    sample_set_list = samples_gcn_df.values.tolist()
                    self.agents[i].prepare_Xs_Y(sample_set_list, self.dic_exp_conf)
            else:
                samples_gcn_df = []
                print("start get samples")
                get_samples_start_time = time.time()
                for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                    sample_set = self.load_sample(i)
                    samples_set_df = pd.DataFrame.from_records(sample_set,columns= ['state','action','next_state','inst_reward','reward','time','generator'])
                    samples_set_df['input'] = samples_set_df[['state','action','next_state','inst_reward','reward']].values.tolist()
                    samples_set_df.drop(['state','action','next_state','inst_reward','reward','time','generator'], axis=1, inplace=True)
                    # samples_set_df['inter_id'] = i
                    samples_gcn_df.append(samples_set_df['input'])
                    if i%100 == 0:
                        print("inter {0} samples_set_df.shape: ".format(i), samples_set_df.shape)
                samples_gcn_df = pd.concat(samples_gcn_df, axis=1)
                print("samples_gcn_df.shape :", samples_gcn_df.shape)
                print("Getting samples time: ", time.time()-get_samples_start_time)

                for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
                        sample_set_list = samples_gcn_df.values.tolist()
                        self.agents[i].prepare_Xs_Y(sample_set_list, self.dic_exp_conf)

        print("------------------Load samples time: ", time.time()-start_time)

    def sample_set_to_sample_gcn_df(self,sample_set):
        print("make results")
        samples_set_df = pd.DataFrame.from_records(sample_set,columns= ['state','action','next_state','inst_reward','reward','time','generator'])
        samples_set_df = samples_set_df.set_index(['time','generator'])
        samples_set_df['input'] = samples_set_df[['state','action','next_state','inst_reward','reward']].values.tolist()
        samples_set_df.drop(['state','action','next_state','inst_reward','reward'], axis=1, inplace=True)
        self.sample_set_list.append(samples_set_df)


    def update_network(self,i):
        print('update agent %d'%i)
        self.agents[i].train_network(self.dic_exp_conf)
        if self.dic_traffic_env_conf["ONE_MODEL"]:
            if self.dic_exp_conf["PRETRAIN"]:
                self.agents[i].q_network.save(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                             "{0}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0]))
                                             )
                shutil.copy(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                         "{0}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0])),
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            elif self.dic_exp_conf["AGGREGATE"]:
                self.agents[i].q_network.save("model/initial", "aggregate.h5")
                shutil.copy("model/initial/aggregate.h5",
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            else:
                self.agents[i].save_network("round_{0}".format(self.cnt_round))

        else:
            if self.dic_exp_conf["PRETRAIN"]:
                self.agents[i].q_network.save(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                             "{0}_inter_{1}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0],
                                                                       self.agents[i].intersection_id))
                                             )
                shutil.copy(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                         "{0}_inter_{1}.h5".format(self.dic_exp_conf["TRAFFIC_FILE"][0],
                                                                   self.agents[i].intersection_id)),
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            elif self.dic_exp_conf["AGGREGATE"]:
                self.agents[i].q_network.save("model/initial", "aggregate_inter_{0}.h5".format(self.agents[i].intersection_id))
                shutil.copy("model/initial/aggregate.h5",
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.h5".format(self.agents[i].intersection_id)))
            else:
                self.agents[i].save_network("round_{0}_inter_{1}".format(self.cnt_round,self.agents[i].intersection_id))

    def update_network_for_agents(self):
        if self.dic_traffic_env_conf["ONE_MODEL"]:
            self.update_network(0)
        else:
            print("update_network_for_agents", self.dic_traffic_env_conf['NUM_AGENTS'])
            for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
                self.update_network(i)





if __name__=="__main__":

    dic_agent_conf = {
    "PRIORITY": True,
    "nan_code":True,
    "att_regularization":False,
    "rularization_rate":0.03,
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    #special care for pretrain
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,

    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
        }

    dic_exp_conf = {
        "RUN_COUNTS": 3600,
        "MODEL_NAME": "STGAT",


        "ROADNET_FILE": "roadnet_{0}.json".format("3_3"),

        "NUM_ROUNDS": 200,
        "NUM_GENERATORS": 4,

        "MODEL_POOL": False,
        "NUM_BEST_MODEL": 3,

        "PRETRAIN_NUM_ROUNDS": 0,
        "PRETRAIN_NUM_GENERATORS": 15,

        "AGGREGATE": False,
        "PRETRAIN": False,
        "DEBUG": False,
        "EARLY_STOP": True
    }

    dic_traffic_env_conf  = {

            "NUM_INTERSECTIONS": 9,
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 10,
            "MIN_ACTION_TIME": 10,
            "DEBUG": False,
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": True,
            'NUM_AGENTS': 1,

            "NEIGHBOR": False,
            "MODEL_NAME": "STGAT",
            "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY":9,




            "SAVEREPLAY": False,
            "NUM_ROW": 3,
            "NUM_COL": 3,

            "VOLUME": 300,
            "ROADNET_FILE": "roadnet_{0}.json".format("3_3"),

            "LIST_STATE_FEATURE": [
                "cur_phase",
                # "time_this_phase",
                # "vehicle_position_img",
                # "vehicle_speed_img",
                # "vehicle_acceleration_img",
                # "vehicle_waiting_time_img",
                "lane_num_vehicle",
                # "lane_num_vehicle_been_stopped_thres01",
                # "lane_num_vehicle_been_stopped_thres1",
                # "lane_queue_length",
                # "lane_num_vehicle_left",
                # "lane_sum_duration_vehicle_left",
                # "lane_sum_waiting_time",
                # "terminal",
                # "coming_vehicle",
                # "leaving_vehicle",
                # "pressure"

                # "adjacency_matrix",
                # "lane_queue_length",
            ],

                "DIC_FEATURE_DIM": dict(
                    D_LANE_QUEUE_LENGTH=(4,),
                    D_LANE_NUM_VEHICLE=(4,),

                    D_COMING_VEHICLE = (12,),
                    D_LEAVING_VEHICLE = (12,),

                    D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                    D_CUR_PHASE=(1,),
                    D_NEXT_PHASE=(1,),
                    D_TIME_THIS_PHASE=(1,),
                    D_TERMINAL=(1,),
                    D_LANE_SUM_WAITING_TIME=(4,),
                    D_VEHICLE_POSITION_IMG=(4, 60,),
                    D_VEHICLE_SPEED_IMG=(4, 60,),
                    D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                    D_PRESSURE=(1,),

                    D_ADJACENCY_MATRIX=(2,),

                ),

            "DIC_REWARD_INFO": {
                "flickering": 0,
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0  # -0.25
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": {
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    3: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                },
                "anon": {
                    # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                    1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    3: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    4: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                    # 'WSWL',
                    # 'ESEL',
                    # 'WSES',
                    # 'NSSS',
                    # 'NSNL',
                    # 'SSSL',
                },
            }
        }

    dic_path= {
            "PATH_TO_MODEL": "/Users/Wingslet/PycharmProjects/RLSignal/model/test/anon_3_3_test",
            "PATH_TO_WORK_DIRECTORY": "records/test/anon_3_3_test",

            "PATH_TO_DATA": "data/test/",
            "PATH_TO_ERROR": "error/test/"
        }

    up = Updater(0, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path)

    up.load_sample_for_agents()

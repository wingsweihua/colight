import config
import copy
from baseline.oneline import OneLine
import os
import time
from multiprocessing import Process
import sys
from script import get_traffic_volume
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='0509_evening_MaxPressure_NewYork_28_7_triple')
    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--env", type=int, default=1)
    parser.add_argument("--gui", type=bool, default=0)
    parser.add_argument("--road_net", type=str, default='28_7')
    parser.add_argument("--volume", type=str, default='newyork')#'300'
    parser.add_argument("--ratio", type=str, default='real_triple')#'0.3_uni'
    parser.add_argument("--model", type=str, default="MaxPressure")#'SlidingFormula','Fixedtime','MaxPressure'
    parser.add_argument("--count",type=int, default=3600)
    parser.add_argument("--lane", type=int, default=3)
    return parser.parse_args()

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i
    return -1

def oneline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    oneline = OneLine(dic_exp_conf=merge(config.DIC_EXP_CONF, dic_exp_conf),
                      dic_agent_conf=merge(getattr(config, "DIC_{0}_AGENT_CONF".format(dic_exp_conf["MODEL_NAME"].upper())),
                                           dic_agent_conf),
                      dic_traffic_env_conf=merge(config.dic_traffic_env_conf, dic_traffic_env_conf),
                      dic_path=merge(config.DIC_PATH, dic_path)
                      )
    oneline.train()
    return

def main(memo, rall, road_net, env, gui, volume, ratio, model,count,lane):
    NUM_COL = int(road_net.split('_')[0])
    NUM_ROW = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL

    ENVIRONMENT = ["sumo", "anon"][env]


    if rall:
        traffic_file_list = [ENVIRONMENT+"_"+road_net+"_%d_%s" %(v,r) for v in [300,500,700] for r in [0.3,0.6]]
    else:
        traffic_file_list=["{0}_{1}_{2}_{3}".format(ENVIRONMENT, road_net, volume, ratio)]

    if env:
        traffic_file_list = [i+ ".json" for i in traffic_file_list ]
    else:
        traffic_file_list = [i+ ".xml" for i in traffic_file_list ]

    process_list = []
    multi_process = True
    n_workers = 60

    # !! important para !!
    # best cycle length sets the min_action_time 1
    # normal formula 1
    min_action_time = 10

    for traffic_file in traffic_file_list:
      #for cl in range(1, 60):
        cl = 30
        model_name = model

        # automatically choose the n_segments
        if "cross" in traffic_file:
            n_segments = 1
        else:
            n_segments = 12
        dic_exp_conf_extra = {
            "RUN_COUNTS": count,
            "MODEL_NAME": model_name,
            "TRAFFIC_FILE": [traffic_file],
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net)
        }
        #
        if volume=='jinan' or volume=='hangzhou' or volume=='newyork' or volume=='LA' or volume=='georgia' or 'chacha' in volume:
            new_volume=volume
            volume=300
            if 'chacha' in new_volume:
                volume=int(new_volume.split('_')[1])
        else:
            new_volume='uniform'
            if 'anon' not in traffic_file:
                volume = int(traffic_file.split('_')[3])
            else:
                volume = get_traffic_volume(traffic_file)

        '''
        ## deeplight agent conf
        dic_agent_conf_extra = {
            "EPOCHS": 100,
            "SAMPLE_SIZE": 300,
            "MAX_MEMORY_LEN": 1000,
            "PHASE_SELECTOR": False,
            "SEPARATE_MEMORY": False,
            "EPSILON": 0

        }
        '''


        dic_traffic_env_conf_extra = {
            "USE_LANE_ADJACENCY": True,
            "NEIGHBOR": False,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",
            "MIN_ACTION_TIME": 20,
            "MEASURE_TIME": 10,
            "IF_GUI": gui,
            "DEBUG": False,
            "TOP_K_ADJACENCY": 4,
            "SIMULATOR_TYPE": ENVIRONMENT,
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": False,
            "ADJACENCY_BY_CONNECTION_OR_GEO":False,
            "TOP_K_ADJACENCY_LANE": 5,
            "MODEL_NAME":model_name,


            "SAVEREPLAY": False,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "LIST_STATE_FEATURE": [
                "cur_phase",
                "time_this_phase",
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
                # "lane_coming_vehicle",
                "coming_vehicle",
                "leaving_vehicle",
                # "pressure",

                # "adjacency_matrix",
                # "lane_queue_length"
            ],

                "DIC_FEATURE_DIM": dict(
                    D_LANE_QUEUE_LENGTH=(4,),
                    D_LANE_NUM_VEHICLE=(4,),

                    D_COMING_VEHICLE = (4,),
                    D_LEAVING_VEHICLE = (4,),

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

                    D_ADJACENCY_MATRIX=(2,)
                ),

            "DIC_REWARD_INFO": {
                "flickering": 0,
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0 # -0.25
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



        ## ==================== multi_phase ====================

        # if dic_traffic_env_conf_extra["LANE_NUM"] == config._LS:
        if lane == 2:
            template = "template_ls"
            fixed_time = [cl, cl, cl, cl]
            fixed_time_dic = config.DIC_FIXEDTIME_MULTI_PHASE
        elif lane == 1:
            template = "template_s"
            fixed_time = [cl, cl]
            fixed_time_dic = config.DIC_FIXEDTIME
            dic_traffic_env_conf_extra["PHASE"] = {
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0], # 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1], # 'NSSS',
                },
                "anon": {
                    1: [0, 1, 0, 1, 0, 0, 0, 0], # 'WSES',
                    2: [0, 0, 0, 0, 0, 1, 0, 1], # 'NSSS',
                },
            }
            dic_traffic_env_conf_extra["LANE_NUM"] = {
                "LEFT": 0,
                "RIGHT": 0,
                "STRAIGHT": 1
            }
        elif lane == 3:
            if new_volume=='jinan':
                template='Jinan'
            elif new_volume=='hangzhou':
                template='Hangzhou'
            elif new_volume=='newyork':
                template='NewYork'
            elif new_volume=='LA':
                template='LA'
            elif new_volume=='georgia':
                template='Georgia'
            elif 'chacha' in new_volume:
                template='Chacha'
            else:
                template = "template_lsr"
            fixed_time_dic = config.DIC_FIXEDTIME_MULTI_PHASE
        else:
            raise ValueError
        lit_template = "lit_" + template

        ## fixedtime agent conf

        dic_agent_conf_extra = {
            "DAY_TIME": 3600,
            "UPDATE_PERIOD": 3600 / n_segments,  # if synthetic, update_period: 3600/12
            "FIXED_TIME": fixed_time_dic[volume],
            "ROUND_UP": 1,
            "PHASE_TO_LANE": [[0, 1], [2, 3]],
            "MIN_PHASE_TIME": 1,
            "TRAFFIC_FILE": [traffic_file],
        }
        dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

        prefix_intersections = str(road_net)
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file+"_"+time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))+"_%d"%cl),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, traffic_file+"_"+time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))+"_%d"%cl),
            "PATH_TO_DATA": os.path.join("data", template, prefix_intersections)
        }

        if multi_process:
            process_list.append(Process(target=oneline_wrapper,
                                        args=(dic_exp_conf_extra, dic_agent_conf_extra,
                                              dic_traffic_env_conf_extra, dic_path_extra))
                                )
        else:
            oneline_wrapper(dic_exp_conf_extra, dic_agent_conf_extra, dic_traffic_env_conf_extra, dic_path_extra)

    if multi_process:
        i = 0
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < n_workers:
                print(i)
                p.start()
                list_cur_p.append(p)
                i += 1
            if len(list_cur_p) < n_workers:
                continue

            idle = check_all_workers_working(list_cur_p)

            while idle == -1:
                time.sleep(1)
                idle = check_all_workers_working(
                    list_cur_p)
            del list_cur_p[idle]

        for p in list_cur_p:
            p.join()

if __name__ == "__main__":
    start=time.time()
    args = parse_args()
    # memo = "test"
    # args.g = True
    main(args.memo, args.all, args.road_net, args.env, args.gui, args.volume, args.ratio,args.model,args.count,args.lane)
    end=time.time()
    duration=end-start
    print('%.2f seconds, %.2f minutes'%(duration,duration/60))

    # def main(memo=None, baseline, road_net, env, gui, volume, ratio, model):

"""
time log for 3600s
Fixedtime: 289.47 seconds, 4.82 min
MaxPressure: 271.12 seconds, 4.51 min
"""
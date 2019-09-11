import json
import os
import pickle
from config import DIC_AGENTS, DIC_ENVS
from copy import deepcopy


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def downsample(path_to_log, i):
    path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
    with open(path_to_pkl, "rb") as f_logging_data:
        logging_data = pickle.load(f_logging_data)
    subset_data = logging_data[::10]
    os.remove(path_to_pkl)
    with open(path_to_pkl, "wb") as f_subset:
        pickle.dump(subset_data, f_subset)

def downsample_for_system(path_to_log,dic_traffic_env_conf):
    for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
        downsample(path_to_log,i)



# TODO test on multiple intersections
def test(model_dir, cnt_round, run_cnt, _dic_traffic_env_conf, if_gui):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d"%cnt_round
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = model_dir
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    with open(os.path.join(records_dir, "exp.conf"), "r") as f:
        dic_exp_conf = json.load(f)

    if os.path.exists(os.path.join(records_dir, "sumo_env.conf")):
        with open(os.path.join(records_dir, "sumo_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    elif os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)

    dic_exp_conf["RUN_COUNTS"] = run_cnt
    dic_traffic_env_conf["IF_GUI"] = if_gui

    # dump dic_exp_conf
    with open(os.path.join(records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)


    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
        dic_agent_conf["MIN_EPSILON"] = 0

    agents = []


    try:
        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](path_to_log=path_to_log,
                                                               path_to_work_directory=dic_path[
                                                                   "PATH_TO_WORK_DIRECTORY"],
                                                               dic_traffic_env_conf=dic_traffic_env_conf)

        done = False
        state = env.reset()

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = dic_exp_conf["MODEL_NAME"]
            if agent_name=='CoLight_Signal':
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=dic_agent_conf,
                    dic_traffic_env_conf=dic_traffic_env_conf,
                    dic_path=dic_path,
                    cnt_round=1,  # useless
                    inter_info=env.list_intersection,
                    intersection_id=str(i)
                )
            else:
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=dic_agent_conf,
                    dic_traffic_env_conf=dic_traffic_env_conf,
                    dic_path=dic_path,
                    cnt_round=1,  # useless
                    intersection_id=str(i)
                )
            agents.append(agent)
            

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            if dic_traffic_env_conf["ONE_MODEL"]:
                agents[i].load_network("{0}".format(model_round))
            else:
                agents[i].load_network("{0}_inter_{1}".format(model_round, agents[i].intersection_id))


        step_num = 0

        attention_dict = {}

        while not done and step_num < int(dic_exp_conf["RUN_COUNTS"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []

            for i in range(dic_traffic_env_conf["NUM_AGENTS"]):

                if "CoLight" in dic_exp_conf["MODEL_NAME"]:
                    one_state = state
                    action_list, attention = agents[i].choose_action(step_num, one_state)
                    cur_time = env.get_current_time()
                    attention_dict[cur_time] = attention
                elif "GCN" in dic_exp_conf["MODEL_NAME"]:
                    one_state = state
                    # print('one_state:',one_state)
                    action_list = agents[i].choose_action(step_num, one_state)
                    # print('action_list:',action_list)
                elif "SimpleDQNOne" in dic_exp_conf["MODEL_NAME"]:
                    one_state = state
                    if True:
                        action_list = agents[i].choose_action(step_num, one_state)
                    else:
                        action_list = agents[i].choose_action_separate(step_num, one_state)
                else:
                    one_state = state[i]
                    action = agents[i].choose_action(step_num, one_state)
                    action_list.append(action)

            next_state, reward, done, _ = env.step(action_list)

            state = next_state
            step_num += 1
        # print('bulk_log_multi_process')
        env.bulk_log_multi_process()
        env.log_attention(attention_dict)

        env.end_sumo()
        if not dic_exp_conf["DEBUG"]:
            path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                       model_round)
            # print("downsample", path_to_log)
            downsample_for_system(path_to_log, dic_traffic_env_conf)
            # print("end down")

    except:
        error_dir = model_dir.replace("model", "errors")
        if os.path.exists(error_dir):
            f = open(os.path.join(error_dir, "error_info.txt"), "a")
            f.write("round_%d fail to test model"%cnt_round)
            f.close()
        else:
            os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info.txt"), "a")
            f.write("round_%d fail to test model"%cnt_round)
            f.close()

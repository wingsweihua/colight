import json
import os
import pandas as pd
import random
import numpy as np
import pickle
from math import isnan
from config import DIC_AGENTS, DIC_ENVS

validation_set = [
    "synthetic-over-WE254-EW221-NS671-SN747-1893.xml",
    # "synthetic-over-WE484-EW484-NS700-SN649-2317.xml",
    # "synthetic-over-WE495-EW511-NS634-SN736-2376.xml",
    "synthetic-over-WE499-EW450-NS502-SN447-1898.xml",
    "synthetic-over-WE510-EW445-NS489-SN524-1968.xml",
    "synthetic-under-WE221-EW300-NS509-SN524-1554.xml",
    "synthetic-under-WE239-EW262-NS690-SN637-1828.xml",
    # "synthetic-under-WE240-EW277-NS509-SN544-1570.xml",
    # "synthetic-under-WE247-EW279-NS232-SN242-1000.xml",
    # "synthetic-under-WE259-EW228-NS265-SN271-1023.xml"
]

DIC_MIN_DURATION = {
    200: 26,
    300: 26,
    350: 27,
    400: 28,
    450: 29,
    500: 30,
    550: 34,
    600: 38,
    650: 40
}


def get_traffic_volume(file_name, run_cnt):
    scale = run_cnt / 3600  # run_cnt > traffic_time, no exact scale
    if "synthetic" in file_name:
        sta = file_name.rfind("-") + 1
        print(file_name, int(int(file_name[sta:-4]) * scale))
        return int(int(file_name[sta:-4]) * scale)
    elif "cross" in file_name:
        sta = file_name.find("equal_") + len("equal_")
        end = file_name.find(".xml")
        return int(int(file_name[sta:end]) * scale * 4)  # lane_num = 4


class ModelPool():
    def __init__(self, dic_path, dic_exp_conf):
        self.dic_path = dic_path
        self.exp_conf = dic_exp_conf

        self.num_best_model = self.exp_conf["NUM_BEST_MODEL"]

        if os.path.exists(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "best_model.pkl")):
            self.best_model_pool = pickle.load(
                open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "best_model.pkl"), "rb"))
        else:
            self.best_model_pool = []

    def single_test(self, cnt_round):
        print("Start testing model pool")

        records_dir = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # run_cnt = 360
        if_gui = False

        nan_thres = 80

        dic_agent_conf = json.load(open(os.path.join(records_dir, "agent.conf"), "r"))
        dic_exp_conf = json.load(open(os.path.join(records_dir, "exp.conf"), "r"))
        dic_traffic_env_conf = json.load(open(os.path.join(records_dir, "traffic_env.conf"), "r"))

        # dic_exp_conf["RUN_COUNTS"] = run_cnt
        run_cnt = dic_exp_conf["RUN_COUNTS"]
        dic_traffic_env_conf["IF_GUI"] = if_gui

        # dump dic_exp_conf
        if os.path.exists(os.path.join(records_dir, "test_exp.conf")):
            json.dump(dic_exp_conf, open(os.path.join(records_dir, "test_exp.conf"), "w"))

        if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
            dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
            dic_agent_conf["MIN_EPSILON"] = 0
        agent_name = dic_exp_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=self.dic_path,
            cnt_round=0,  # useless
        )
        # try:
        if 1:
            # test
            agent.load_network("round_{0}".format(cnt_round))

            path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                       "round_{0}".format(cnt_round))
            if not os.path.exists(path_to_log):
                os.makedirs(path_to_log)
            env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                             path_to_log=path_to_log,
                             path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
                             dic_traffic_env_conf=dic_traffic_env_conf)

            done = False
            state = env.reset()
            step_num = 0

            while not done and step_num < int(dic_exp_conf["RUN_COUNTS"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):
                action_list = []
                for one_state in state:
                    action = agent.choose_action(step_num, one_state)

                    action_list.append(action)

                next_state, reward, done, _ = env.step(action_list)

                state = next_state
                step_num += 1
            env.bulk_log()
            env.end_sumo()

            # summary items (duration) from csv
            df_vehicle_inter_0 = pd.read_csv(os.path.join(path_to_log, "vehicle_inter_0.csv"),
                                             sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                             names=["vehicle_id", "enter_time", "leave_time"])

            duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values

            dur = np.mean([time for time in duration if not isnan(time)])
            real_traffic_vol = 0
            nan_num = 0
            for time in duration:
                if not isnan(time):
                    real_traffic_vol += 1
                else:
                    nan_num += 1

            traffic_vol = get_traffic_volume(dic_exp_conf["TRAFFIC_FILE"][0], run_cnt)

            print(nan_num, nan_thres, self.best_model_pool)
            if nan_num < nan_thres:
                cnt = 0
                for i in range(len(self.best_model_pool)):
                    if self.best_model_pool[i][1] > dur:
                        break
                    cnt += 1

                self.best_model_pool.insert(cnt, [cnt_round, dur])

                num_max = min(len(self.best_model_pool), self.exp_conf["NUM_BEST_MODEL"])
                self.best_model_pool = self.best_model_pool[:num_max]

                # log best models through rounds
                print(self.best_model_pool)
                f = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "best_model_pool.log"), "a")
                f.write("round: %d " % cnt_round)
                for i in range(len(self.best_model_pool)):
                    f.write("id: %d, duration: %f, " % (self.best_model_pool[i][0], self.best_model_pool[i][1]))
                f.write("\n")
                f.close()
            print("model pool ends")
        # except:
        #    print("fail to test model:%s"%model_round)
        #    pass

    def model_compare(self, cnt_round):
        print("Start testing model pool")

        records_dir = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # run_cnt = 360
        if_gui = False

        nan_thres = 80

        dic_agent_conf = json.load(open(os.path.join(records_dir, "agent.conf"), "r"))
        dic_exp_conf = json.load(open(os.path.join(records_dir, "exp.conf"), "r"))
        dic_sumo_env_conf = json.load(open(os.path.join(records_dir, "sumo_env.conf"), "r"))

        # dic_exp_conf["RUN_COUNTS"] = run_cnt
        run_cnt = dic_exp_conf["RUN_COUNTS"]
        dic_sumo_env_conf["IF_GUI"] = if_gui

        # dump dic_exp_conf
        if os.path.exists(os.path.join(records_dir, "test_exp.conf")):
            json.dump(dic_exp_conf, open(os.path.join(records_dir, "test_exp.conf"), "w"))


        # try:
        path_to_log = os.path.join(records_dir, "test_round", "round_%d"%cnt_round)
        if 1:
            # summary items (duration) from csv
            df_vehicle_inter_0 = pd.read_csv(os.path.join(path_to_log, "vehicle_inter_0.csv"),
                                             sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                             names=["vehicle_id", "enter_time", "leave_time"])

            duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values

            dur = np.mean([time for time in duration if not isnan(time)])
            real_traffic_vol = 0
            nan_num = 0
            for time in duration:
                if not isnan(time):
                    real_traffic_vol += 1
                else:
                    nan_num += 1

            traffic_vol = get_traffic_volume(dic_exp_conf["TRAFFIC_FILE"][0], run_cnt)

            print(nan_num, nan_thres, self.best_model_pool)
            if nan_num < nan_thres:
                cnt = 0
                for i in range(len(self.best_model_pool)):
                    if self.best_model_pool[i][1] > dur:
                        break
                    cnt += 1

                self.best_model_pool.insert(cnt, [cnt_round, dur])

                num_max = min(len(self.best_model_pool), self.exp_conf["NUM_BEST_MODEL"])
                self.best_model_pool = self.best_model_pool[:num_max]

                # log best models through rounds
                print(self.best_model_pool)
                f = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "best_model_pool.log"), "a")
                f.write("round: %d " % cnt_round)
                for i in range(len(self.best_model_pool)):
                    f.write("id: %d, duration: %f, " % (self.best_model_pool[i][0], self.best_model_pool[i][1]))
                f.write("\n")
                f.close()
            print("model pool ends")

    def get(self):
        if not self.best_model_pool:
            return
        else:
            ind = random.randint(0, len(self.best_model_pool) - 1)
            return self.best_model_pool[ind][0]

    def dump_model_pool(self):
        if self.best_model_pool:
            pickle.dump(self.best_model_pool,
                        open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "best_model.pkl"), "wb"))

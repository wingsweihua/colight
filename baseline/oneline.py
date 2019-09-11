
import json
import os
import shutil
import xml.etree.ElementTree as ET
import pickle
import time

from config import DIC_AGENTS, DIC_ENVS

def downsample(path_to_log):

    path_to_pkl = os.path.join(path_to_log, "inter_0.pkl")
    f_logging_data = open(path_to_pkl, "rb")
    logging_data = pickle.load(f_logging_data)
    subset_data = logging_data[::10]
    f_logging_data.close()
    os.remove(path_to_pkl)
    f_subset = open(path_to_pkl, "wb")
    pickle.dump(subset_data, f_subset)
    f_subset.close()


class OneLine:

    _LIST_SUMO_FILES = [
        "cross.tll.xml",
        "cross.car.type.xml",
        "cross.con.xml",
        "cross.edg.xml",
        "cross.net.xml",
        "cross.netccfg",
        "cross.nod.xml",
        "cross.sumocfg",
        "cross.typ.xml"
    ]

    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg
        sumo_cfg = ET.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(sumo_config_file_output_name)

    def _path_check(self):
        # check path
        if os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            if self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_MODEL"]):
            if self.dic_path["PATH_TO_MODEL"] != "model/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_MODEL"])

    def _copy_conf_file(self, path=None):
        # write conf files
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_sumo_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files
        for file_name in self._LIST_SUMO_FILES:
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], file_name),
                        os.path.join(path, file_name))
        for file_name in self.dic_exp_conf["TRAFFIC_FILE"]:
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], file_name),
                        os.path.join(path, file_name))

    def _copy_anon_file(self, path=None):
        # hard code !!!
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files

        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["TRAFFIC_FILE"][0]),
                        os.path.join(path, self.dic_exp_conf["TRAFFIC_FILE"][0]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_exp_conf["ROADNET_FILE"]))

    def _modify_sumo_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # modify sumo files
        if self.dic_exp_conf["MULTI_TRAFFIC"]:
            self._set_traffic_file(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "cross.sumocfg"),
                                   os.path.join(path, "cross.sumocfg"),
                                   [self.dic_exp_conf["TRAFFIC_FILE"][0]])
        else:
            self._set_traffic_file(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "cross.sumocfg"),
                                   os.path.join(path, "cross.sumocfg"),
                                   self.dic_exp_conf["TRAFFIC_FILE"])

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []

        # do file operations
        self._path_check()
        self._copy_conf_file()
        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'sumo':
            self._copy_sumo_file()
            self._modify_sumo_file()
        elif self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'anon':
            self._copy_anon_file()

        if self.dic_exp_conf["MODEL_NAME"] == "Deeplight":
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["EPSILON"]

        for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
            agent_name = self.dic_exp_conf["MODEL_NAME"]
            if self.dic_exp_conf["MODEL_NAME"] in ['Fixedtime','SOTL','SlidingFormula']:
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=dic_agent_conf,
                    dic_traffic_env_conf=dic_traffic_env_conf,
                    dic_path=dic_path,
                    cnt_round=0
                )
            elif self.dic_exp_conf["MODEL_NAME"]=='MaxPressure':
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=dic_agent_conf,
                    dic_traffic_env_conf=dic_traffic_env_conf,
                    dic_path=dic_path,
                    cnt_round=0,
                    intersection_id = str(i)
                )
            self.agents.append(agent)



        self.env = DIC_ENVS[self.dic_traffic_env_conf["SIMULATOR_TYPE"]](
                           path_to_log = self.dic_path["PATH_TO_WORK_DIRECTORY"],
                           path_to_work_directory = self.dic_path["PATH_TO_WORK_DIRECTORY"],
                           dic_traffic_env_conf = self.dic_traffic_env_conf)



    def train(self):
        print("start train")
        total_run_cnt = self.dic_exp_conf["RUN_COUNTS"]

        # initialize output streams
        file_name_memory = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "memories.txt")

        done = False
        state = self.env.reset()
        step_num = 0
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        if self.dic_traffic_env_conf['DEBUG']:
                start_time = time.time()
        while not done and current_time < total_run_cnt:

            action_list = []

            for i in range(len(state)):
                one_state = state[i]

                action = self.agents[i].choose_action(step_num, one_state)

                action_list.append(action)

            next_state, reward, done, _ = self.env.step(action_list)

            # debug
            f_memory = open(file_name_memory, "a")
            # output to std out and file
            memory_str = 'time = {0}\taction = {1}\tcurrent_phase = {2}\treward = {3}' \
                         .format(current_time, str(action_list),
                            str([state[i]["cur_phase"][0] for i in range(len(state))]),
                            str(reward),
                            )
            f_memory.write(memory_str + "\n")
            f_memory.close()
            current_time = self.env.get_current_time()  # in seconds

            # remember
            if self.dic_exp_conf["MODEL_NAME"] == "Deeplight":
                for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                    self.agent[i].remember(state[0], action, reward[0], next_state[0])
                    self.agent[i].update_network(False, False, current_time)
                    self.agent[i].update_network_bar()

            state = next_state
            step_num += 1

        if self.dic_traffic_env_conf['DEBUG']:
                print("Training time: ",time.time()-start_time)


        if self.dic_traffic_env_conf['DEBUG']:
                start_time = time.time()
        self.env.bulk_log_multi_process()
        if self.dic_traffic_env_conf['DEBUG']:
                print("Log time: ",time.time()-start_time)

        if self.dic_traffic_env_conf['DEBUG']:
                start_time = time.time()
        downsample(self.dic_path["PATH_TO_WORK_DIRECTORY"])
        if self.dic_traffic_env_conf['DEBUG']:
                print("downsample time: ",time.time()-start_time)
        print("Training END")

    def test(self):

        pass

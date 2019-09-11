import json
import os
import shutil
import xml.etree.ElementTree as ET
from generator import Generator
from construct_sample import ConstructSample
from updater import Updater
from multiprocessing import Process, Pool
from model_pool import ModelPool
import random
import pickle
import model_test
import pandas as pd
import numpy as np
from math import isnan
import sys
import time
import traceback

class Pipeline:
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

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_MODEL"]):
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_MODEL"])

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
        self._set_traffic_file(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "cross.sumocfg"),
                               os.path.join(path, "cross.sumocfg"),
                               self.dic_exp_conf["TRAFFIC_FILE"])

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        # do file operations
        self._path_check()
        self._copy_conf_file()
        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'sumo':
            self._copy_sumo_file()
            self._modify_sumo_file()
        elif self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'anon':
            self._copy_anon_file()
        # test_duration
        self.test_duration = []

        sample_num = 10 if self.dic_traffic_env_conf["NUM_INTERSECTIONS"]>=10 else min(self.dic_traffic_env_conf["NUM_INTERSECTIONS"], 9)
        print("sample_num for early stopping:", sample_num)
        self.sample_inter_id = random.sample(range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]), sample_num)


    def early_stopping(self, dic_path, cnt_round): # Todo multi-process
        print("decide whether to stop")
        early_stopping_start_time = time.time()
        record_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", "round_"+str(cnt_round))

        ave_duration_all = []
        # compute duration
        for inter_id in self.sample_inter_id:
            try:
                df_vehicle_inter_0 = pd.read_csv(os.path.join(record_dir, "vehicle_inter_{0}.csv".format(inter_id)),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])
                duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
                ave_duration = np.mean([time for time in duration if not isnan(time)])
                ave_duration_all.append(ave_duration)
            except FileNotFoundError:
                error_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
                if not os.path.exists(error_dir):
                    os.makedirs(error_dir)
                f = open(os.path.join(error_dir, "error_info.txt"), "a")
                f.write("Fail to read csv of inter {0} in early stopping of round {1}\n".format(inter_id, cnt_round))
                f.close()
                pass

        ave_duration = np.mean(ave_duration_all)
        self.test_duration.append(ave_duration)
        early_stopping_end_time = time.time()
        print("early_stopping time: {0}".format(early_stopping_end_time - early_stopping_start_time) )
        if len(self.test_duration) < 30:
            return 0
        else:
            duration_under_exam = np.array(self.test_duration[-15:])
            mean_duration = np.mean(duration_under_exam)
            std_duration = np.std(duration_under_exam)
            max_duration = np.max(duration_under_exam)
            if std_duration/mean_duration < 0.1 and max_duration < 1.5 * mean_duration:
                return 1
            else:
                return 0



    def generator_wrapper(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          best_round=None):
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              best_round=best_round
                              )
        print("make generator")
        generator.generate()
        print("generator_wrapper end")
        return

    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, best_round=None, bar_round=None):

        updater = Updater(
            cnt_round=cnt_round,
            dic_agent_conf=dic_agent_conf,
            dic_exp_conf=dic_exp_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            best_round=best_round,
            bar_round=bar_round
        ) 

        updater.load_sample_for_agents()
        updater.update_network_for_agents()
        print("updater_wrapper end")
        return

    def model_pool_wrapper(self, dic_path, dic_exp_conf, cnt_round):
        model_pool = ModelPool(dic_path, dic_exp_conf)
        model_pool.model_compare(cnt_round)
        model_pool.dump_model_pool()


        return
        #self.best_round = model_pool.get()
        #print("self.best_round", self.best_round)

    def downsample(self, path_to_log, i):

        path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
        with open(path_to_pkl, "rb") as f_logging_data:
            try:
                logging_data = pickle.load(f_logging_data)
                subset_data = logging_data[::10]
                print(subset_data)
                os.remove(path_to_pkl)
                with open(path_to_pkl, "wb") as f_subset:
                    try:
                        pickle.dump(subset_data, f_subset)
                    except Exception as e:
                        print("----------------------------")
                        print("Error occurs when WRITING pickles when down sampling for inter {0}".format(i))
                        print('traceback.format_exc():\n%s' % traceback.format_exc())
                        print("----------------------------")

            except Exception as e:
                # print("CANNOT READ %s"%path_to_pkl)
                print("----------------------------")
                print("Error occurs when READING pickles when down sampling for inter {0}, {1}".format(i, f_logging_data))
                print('traceback.format_exc():\n%s' % traceback.format_exc())
                print("----------------------------")


    def downsample_for_system(self, path_to_log, dic_traffic_env_conf):
        for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.downsample(path_to_log, i)

    def construct_sample_multi_process(self, train_round, cnt_round, batch_size=200):
        cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                             dic_traffic_env_conf=self.dic_traffic_env_conf)
        if batch_size > self.dic_traffic_env_conf['NUM_INTERSECTIONS']:
            batch_size_run = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, self.dic_traffic_env_conf['NUM_INTERSECTIONS'], batch_size_run):
            start = batch
            stop = min(batch + batch_size, self.dic_traffic_env_conf['NUM_INTERSECTIONS'])
            process_list.append(Process(target=self.construct_sample_batch, args=(cs, start, stop)))

        for t in process_list:
            t.start()
        for t in process_list:
            t.join()

    def construct_sample_batch(self, cs, start,stop):
        for inter_id in range(start, stop):
            print("make construct_sample_wrapper for ", inter_id)
            cs.make_reward(inter_id)
        

    def run(self, multi_process=False):

        best_round, bar_round = None, None

        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"w")
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()

        if self.dic_exp_conf["PRETRAIN"]:
            if os.listdir(self.dic_path["PATH_TO_PRETRAIN_MODEL"]): 
                for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
                    #TODO:only suitable for CoLight
                    shutil.copy(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                            "round_0_inter_%d.h5" % i),
                                os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_%d.h5"%i))
            else:
                if not os.listdir(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
                    for cnt_round in range(self.dic_exp_conf["PRETRAIN_NUM_ROUNDS"]):
                        print("round %d starts" % cnt_round)

                        process_list = []

                        # ==============  generator =============
                        if multi_process:
                            for cnt_gen in range(self.dic_exp_conf["PRETRAIN_NUM_GENERATORS"]):
                                p = Process(target=self.generator_wrapper,
                                            args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                                  self.dic_agent_conf, self.dic_traffic_env_conf, best_round)
                                            )
                                print("before")
                                p.start()
                                print("end")
                                process_list.append(p)
                            print("before join")
                            for p in process_list:
                                p.join()
                            print("end join")
                        else:
                            for cnt_gen in range(self.dic_exp_conf["PRETRAIN_NUM_GENERATORS"]):
                                self.generator_wrapper(cnt_round=cnt_round,
                                                       cnt_gen=cnt_gen,
                                                       dic_path=self.dic_path,
                                                       dic_exp_conf=self.dic_exp_conf,
                                                       dic_agent_conf=self.dic_agent_conf,
                                                       dic_traffic_env_conf=self.dic_traffic_env_conf,
                                                       best_round=best_round)

                        # ==============  make samples =============
                        # make samples and determine which samples are good

                        train_round = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"], "train_round")
                        if not os.path.exists(train_round):
                            os.makedirs(train_round)
                        cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                             dic_traffic_env_conf=self.dic_traffic_env_conf)
                        cs.make_reward()

                if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                    if multi_process:
                        p = Process(target=self.updater_wrapper,
                                    args=(0,
                                          self.dic_agent_conf,
                                          self.dic_exp_conf,
                                          self.dic_traffic_env_conf,
                                          self.dic_path,
                                          best_round))
                        p.start()
                        p.join()
                    else:
                        self.updater_wrapper(cnt_round=0,
                                             dic_agent_conf=self.dic_agent_conf,
                                             dic_exp_conf=self.dic_exp_conf,
                                             dic_traffic_env_conf=self.dic_traffic_env_conf,
                                             dic_path=self.dic_path,
                                             best_round=best_round)
        # train with aggregate samples
        if self.dic_exp_conf["AGGREGATE"]:
            if "aggregate.h5" in os.listdir("model/initial"):
                shutil.copy("model/initial/aggregate.h5",
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            else:
                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(0,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path,
                                      best_round))
                    p.start()
                    p.join()
                else:
                    self.updater_wrapper(cnt_round=0,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_traffic_env_conf=self.dic_traffic_env_conf,
                                         dic_path=self.dic_path,
                                         best_round=best_round)

        self.dic_exp_conf["PRETRAIN"] = False
        self.dic_exp_conf["AGGREGATE"] = False

        # trainf
        for cnt_round in range(self.dic_exp_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            round_start_time = time.time()

            process_list = []

            print("==============  generator =============")
            generator_start_time = time.time()
            if multi_process:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    p = Process(target=self.generator_wrapper,
                                args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                      self.dic_agent_conf, self.dic_traffic_env_conf, best_round)
                                )
                    print("before p")
                    p.start()
                    print("end p")
                    process_list.append(p)
                print("before join")
                for i in range(len(process_list)):
                    p = process_list[i]
                    print("generator %d to join" % i)
                    p.join()
                    print("generator %d finish join" % i)
                print("end join")
            else:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    self.generator_wrapper(cnt_round=cnt_round,
                                           cnt_gen=cnt_gen,
                                           dic_path=self.dic_path,
                                           dic_exp_conf=self.dic_exp_conf,
                                           dic_agent_conf=self.dic_agent_conf,
                                           dic_traffic_env_conf=self.dic_traffic_env_conf,
                                           best_round=best_round)
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time
            print("==============  make samples =============")
            # make samples and determine which samples are good
            making_samples_start_time = time.time()

            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(train_round):
                os.makedirs(train_round)

            cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)
            cs.make_reward_for_system()


            # EvaluateSample()
            making_samples_end_time = time.time()
            making_samples_total_time = making_samples_end_time - making_samples_start_time

            print("==============  update network =============")
            update_network_start_time = time.time()
            if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(cnt_round,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path,
                                      best_round,
                                      bar_round))
                    p.start()
                    print("update to join")
                    p.join()
                    print("update finish join")
                else:
                    self.updater_wrapper(cnt_round=cnt_round,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_traffic_env_conf=self.dic_traffic_env_conf,
                                         dic_path=self.dic_path,
                                         best_round=best_round,
                                         bar_round=bar_round)

            if not self.dic_exp_conf["DEBUG"]:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                               "round_" + str(cnt_round), "generator_" + str(cnt_gen))
                    try:
                        self.downsample_for_system(path_to_log,self.dic_traffic_env_conf)
                    except Exception as e:
                        print("----------------------------")
                        print("Error occurs when downsampling for round {0} generator {1}".format(cnt_round, cnt_gen))
                        print("traceback.format_exc():\n%s"%traceback.format_exc())
                        print("----------------------------")
            update_network_end_time = time.time()
            update_network_total_time = update_network_end_time - update_network_start_time

            print("==============  test evaluation =============")
            test_evaluation_start_time = time.time()
            if multi_process:
                p = Process(target=model_test.test,
                            args=(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf, False))
                p.start()
                if self.dic_exp_conf["EARLY_STOP"]:
                    p.join()
            else:
                model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf, if_gui=False)

            test_evaluation_end_time = time.time()
            test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time

            print('==============  early stopping =============')
            if self.dic_exp_conf["EARLY_STOP"]:
                flag = self.early_stopping(self.dic_path, cnt_round)
                if flag == 1:
                    print("early stopping!")
                    print("training ends at round %s" % cnt_round)
                    break

            print('==============  model pool evaluation =============')
            if self.dic_exp_conf["MODEL_POOL"] and cnt_round > 50:
                if multi_process:
                    p = Process(target=self.model_pool_wrapper,
                                args=(self.dic_path,
                                      self.dic_exp_conf,
                                      cnt_round),
                                )
                    p.start()
                    print("model_pool to join")
                    p.join()
                    print("model_pool finish join")
                else:
                    self.model_pool_wrapper(dic_path=self.dic_path,
                                            dic_exp_conf=self.dic_exp_conf,
                                            cnt_round=cnt_round)
                model_pool_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "best_model.pkl")
                if os.path.exists(model_pool_dir):
                    model_pool = pickle.load(open(model_pool_dir, "rb"))
                    ind = random.randint(0, len(model_pool) - 1)
                    best_round = model_pool[ind][0]
                    ind_bar = random.randint(0, len(model_pool) - 1)
                    flag = 0
                    while ind_bar == ind and flag < 10:
                        ind_bar = random.randint(0, len(model_pool) - 1)
                        flag += 1
                    # bar_round = model_pool[ind_bar][0]
                    bar_round = None
                else:
                    best_round = None
                    bar_round = None

                # downsample
                if not self.dic_exp_conf["DEBUG"]:
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                               "round_" + str(cnt_round))
                    self.downsample_for_system(path_to_log, self.dic_traffic_env_conf)
            else:
                best_round = None

            print("best_round: ", best_round)

            print("Generator time: ",generator_total_time)
            print("Making samples time:", making_samples_total_time)
            print("update_network time:", update_network_total_time)
            print("test_evaluation time:", test_evaluation_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"running_time.csv"),"a")
            f_time.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(generator_total_time,making_samples_total_time,
                                                          update_network_total_time,test_evaluation_total_time,
                                                          time.time()-round_start_time))
            f_time.close()



import sys
sys.path.append("..")
import numpy as np
import os
from agent import Agent
import xml.etree.ElementTree as ET
from math import ceil, floor

class FormulaAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round):

        super(FormulaAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path)

        self.n_segments = int(ceil(self.dic_agent_conf["DAY_TIME"]/self.dic_agent_conf["UPDATE_PERIOD"]))


        self.list_vehicles_count_for_phase = np.zeros((self.n_segments, 4))
        self.list_fixed_time = np.zeros((self.n_segments, 2))

        # the traffic record can only record the first day
        traffic_xml = ET.parse(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], self.dic_agent_conf["TRAFFIC_FILE"][0]))
        routes = traffic_xml.getroot()
        for vehicle in routes:
            veh_dic = vehicle.attrib
            ts = int(veh_dic["begin"])
            if ts < self.dic_agent_conf["DAY_TIME"]:
                if veh_dic["departPos"] == "1":
                    self.list_vehicles_count_for_phase[
                        int(floor((ts % self.dic_agent_conf["DAY_TIME"]) / self.dic_agent_conf["UPDATE_PERIOD"])), 0] = int(veh_dic["vehsPerHour"])
                elif veh_dic["departPos"] == "2":
                    self.list_vehicles_count_for_phase[
                        int(floor((ts % self.dic_agent_conf["DAY_TIME"]) / self.dic_agent_conf["UPDATE_PERIOD"])), 1] = int(veh_dic["vehsPerHour"])
                elif veh_dic["departPos"] == "3":
                    self.list_vehicles_count_for_phase[
                        int(floor((ts % self.dic_agent_conf["DAY_TIME"]) / self.dic_agent_conf["UPDATE_PERIOD"])), 2] = int(veh_dic["vehsPerHour"])
                elif veh_dic["departPos"] == "4":
                    self.list_vehicles_count_for_phase[
                        int(floor((ts % self.dic_agent_conf["DAY_TIME"]) / self.dic_agent_conf["UPDATE_PERIOD"])), 3] = int(veh_dic["vehsPerHour"])
            else:
                break

        # change to vec/h
        self.list_vehicles_count_for_phase = self.list_vehicles_count_for_phase #/self.dic_agent_conf["UPDATE_PERIOD"]*3600
        # change vec count to per lane
        self.list_vehicles_count_for_phase = self.list_vehicles_count_for_phase / self.dic_traffic_env_conf["NUM_LANES"]

        for i, vehicles_count_for_phase in enumerate(self.list_vehicles_count_for_phase):
            self.list_fixed_time[i] = self.get_phase_split(vehicles_count_for_phase)


        if self.n_segments == 1: # means uniform, and execute the best cycle length in grid search
            self.list_fixed_time = [self.dic_agent_conf["FIXED_TIME"]]
        self.FIXED_TIME = self.list_fixed_time[0]

        # debug
        _path = os.path.split(self.dic_path["PATH_TO_WORK_DIRECTORY"])
        memo = _path[0].replace("records", "summary")
        if not os.path.exists(memo):
            os.makedirs(memo)
        f = open(os.path.join(memo, "cycle_length.txt"), 'a')
        f.write(self.dic_agent_conf["TRAFFIC_FILE"][0] + "\t" + repr(self.list_fixed_time) + '\n')
        f.close()

    def round_up(self, x, b=5):
        round_x = (b * np.ceil(x.astype(float) / b)).astype(int)
        round_x[np.where(round_x < self.dic_agent_conf["MIN_PHASE_TIME"])] = self.dic_agent_conf["MIN_PHASE_TIME"]
        return round_x

    def choose_action(self, count, state):

        ''' choose the best action for current state '''

        q_values = np.array([0, 0])
        if state["time_this_phase"][0] <= self.FIXED_TIME[state["cur_phase"][0]]:
            self.action = state["cur_phase"][0]
        else:
            self.action = (self.action + 1)%self.dic_traffic_env_conf["NUM_PHASES"]
            self.FIXED_TIME = self.list_fixed_time[
                int(floor((count%self.dic_agent_conf["DAY_TIME"])/self.dic_agent_conf["UPDATE_PERIOD"]))]
        return self.action

    def get_phase_split(self, vehicles_count_for_phase):

        h = 2.45
        tL_set = 5
        tL = 7
        # not very robust to discriminate uniform and new synthetic data
        if self.n_segments == 1:
            PHF = 1
            vc = 1
        else:
            PHF = 1
            vc = 1
        N = self.dic_sumo_env_conf["NUM_PHASES"]

        vehicles_count_for_critical_lane_phase = self.get_vehicles_count_for_critical_lane_phase(
            vehicles_count_for_phase)

        max_allowed_vol = 3600 / h * PHF * vc
        total_vol = np.sum(vehicles_count_for_critical_lane_phase)
        if total_vol/max_allowed_vol > 0.95:
            cycle_length = N * tL / (1 - 0.95)
        else:
            cycle_length = N * tL / (1 - total_vol / max_allowed_vol)

        if cycle_length < 0:
            sys.exit("cycle length calculation error")

        effect_cycle_length = cycle_length - tL_set * N

        if np.sum(vehicles_count_for_critical_lane_phase) != 0:
            phase_split = np.copy(vehicles_count_for_critical_lane_phase)\
                          / np.sum(vehicles_count_for_critical_lane_phase) \
                          * effect_cycle_length
        else:
            phase_split = np.full(shape=(len(vehicles_count_for_critical_lane_phase),),
                                  fill_value=1/len(vehicles_count_for_critical_lane_phase)) \
                          * effect_cycle_length

        phase_split = self.round_up(phase_split, b=self.dic_agent_conf["ROUND_UP"])

        return phase_split

    def get_vehicles_count_for_critical_lane_phase(self, vehicles_count_for_lane):

        list_phase_critical_lane_volume = []

        for phase_lanes in self.dic_agent_conf["PHASE_TO_LANE"]:
            list_phase_critical_lane_volume.append(np.max(vehicles_count_for_lane[phase_lanes]))

        return list_phase_critical_lane_volume






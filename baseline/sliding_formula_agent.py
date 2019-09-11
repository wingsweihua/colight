import sys
sys.path.append("..")
import numpy as np
import os
from agent import Agent
import xml.etree.ElementTree as ET
from math import ceil, floor
import json

class SlidingFormulaAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round):

        super(SlidingFormulaAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path)

        self.FIXED_TIME = np.full((4,), 30)

    def round_up(self, x, b=5):
        round_x = (b * np.ceil(x.astype(float) / b)).astype(int)
        round_x[np.where(round_x < self.dic_agent_conf["MIN_PHASE_TIME"])] = self.dic_agent_conf["MIN_PHASE_TIME"]
        return round_x

    def choose_action(self, count, state):

        ''' choose the best action for current state '''

        q_values = np.array([0, 0])
        print(">> ", state["cur_phase"][0])

        if self.dic_traffic_env_conf['SIMULATOR_TYPE'] == 'anon':
            if state["time_this_phase"][0] <= self.FIXED_TIME[state["cur_phase"][0]-1]:
                self.action = state["cur_phase"][0]-1
            else:
                self.action = (self.action + 1) % 4
                if self.action == 0:
                    # change the FIXED_TIME
                    lane_coming_vehicle = state['lane_coming_vehicle'] /self.dic_agent_conf["UPDATE_PERIOD"] *3600
                    self.FIXED_TIME = self.get_phase_split(lane_coming_vehicle)

                    # _path = os.path.split(self.dic_path["PATH_TO_WORK_DIRECTORY"])
                    # memo = _path[0].replace("records", "summary")
                    # if not os.path.exists(memo):
                    #     os.makedirs(memo)
                    # f = open(os.path.join(memo, "cycle_length.txt"), 'a')
                    # f.write(self.dic_agent_conf["TRAFFIC_FILE"] + "\t" + repr(self.FIXED_TIME) + '\n')
                    # f.close()
        else:
            if state["time_this_phase"][0] <= self.FIXED_TIME[state["cur_phase"][0]]:
                self.action = state["cur_phase"][0]
            else:
                self.action = (self.action + 1) % 4
                self.FIXED_TIME = self.list_fixed_time[
                    int(floor((count%self.dic_agent_conf["DAY_TIME"])/self.dic_agent_conf["UPDATE_PERIOD"]))]
        return self.action

    def get_phase_split(self, vehicles_count_for_phase):

        h = 2.45
        tL_set = 5
        tL = 7

        PHF = 1
        vc = 1

        N = 4#len(self.dic_traffic_env_conf["PHASE"])

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







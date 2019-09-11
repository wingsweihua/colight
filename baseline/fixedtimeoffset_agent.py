import sys
sys.path.append("..")
from agent import Agent
import random
import numpy as np


class FixedtimeOffsetAgent(Agent):


    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round):

        super(FixedtimeOffsetAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path)

        self.current_phase_time = 0

        traffic_demand = self.dic_traffic_env_conf['TRAFFIC_FILE'].split('_')[3]
        if self.dic_traffic_env_conf['SIMULATOR_TYPE'] == 'sumo':
            ratio = self.dic_traffic_env_conf['TRAFFIC_FILE'].split('_')[4].split('.xml')[0]
        else:
            ratio = self.dic_traffic_env_conf['TRAFFIC_FILE'].split('_')[4].split('.json')[0]

        traffic_demand = int(traffic_demand)
        ratio = float(ratio)
        self.dic_agent_conf["FIXED_TIME"] = self.get_phase_split(traffic_demand,ratio)

        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == "anon":
            self.DIC_PHASE_MAP = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                0: 0
            }
        else:
            self.DIC_PHASE_MAP = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                -1: -1
            }


    def choose_action(self, count, state):
        ''' choose the best action for current state '''

        if state["cur_phase"][0] == -1:
            return self.action
        cur_phase = self.DIC_PHASE_MAP[state["cur_phase"][0]]

        if self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
            if count < 30 + self.dic_agent_conf["FIXED_TIME"][cur_phase]:
                self.action = cur_phase
                self.current_phase_time += 1
                return cur_phase

            elif state["time_this_phase"][0] <= self.dic_agent_conf["FIXED_TIME"][cur_phase]:
                self.action = cur_phase
                self.current_phase_time += 1
                return cur_phase

            else:
                self.current_phase_time = 0
                self.action = (cur_phase + 1) % len(self.dic_traffic_env_conf["PHASE"])
                return (cur_phase + 1) % len(self.dic_traffic_env_conf["PHASE"])

        else:
            if state["time_this_phase"][0] >= self.dic_agent_conf["FIXED_TIME"][cur_phase] and cur_phase != -1:
                self.current_phase_time = 0
                self.action = 1
                return 1
            else:
                self.current_phase_time += 1
                self.action = 0
                return 0


    def get_phase_split(self, traffic_demand, ratio):

        h = 2.45
        tL_set = 5
        tL = 7
        PHF = 1
        vc = 1
        N = 2
        vehicles_count_for_critical_lane_phase = traffic_demand*(1+ratio)
        max_allowed_vol = 3600 / h * PHF * vc
        total_vol = np.sum(vehicles_count_for_critical_lane_phase)
        if total_vol / max_allowed_vol > 0.95:
            cycle_length = N * tL / (1 - 0.95)
        else:
            cycle_length = N * tL / (1 - total_vol / max_allowed_vol)

        if cycle_length < 0:
            sys.exit("cycle length calculation error")

        effect_cycle_length = cycle_length - tL_set * N
        if np.sum(vehicles_count_for_critical_lane_phase) != 0:
            phase_split = np.copy(vehicles_count_for_critical_lane_phase)/np.sum(vehicles_count_for_critical_lane_phase) * effect_cycle_length
        else:
            phase_split = np.full(shape=(len(vehicles_count_for_critical_lane_phase),), fill_value=1/len(vehicles_count_for_critical_lane_phase)) * effect_cycle_length

        phase_split = int(phase_split)+1
        green =int (phase_split/(1+ratio))+1
        red = int(phase_split/(1+ratio)*ratio)+1


        while green%5!=0:
            green +=1

        while red%5!=0:
            red+=1

        return [green,red]
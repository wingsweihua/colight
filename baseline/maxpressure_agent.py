import sys

sys.path.append("..")
from agent import Agent
import random
import numpy as np


ACTUATED = True


class MaxPressureAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):

        super(MaxPressureAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.current_phase_time = 0
        ### Arterial
        self.phase_length = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf["SIMULATOR_TYPE"]])

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

        phase_1 = max((np.sum(state['coming_vehicle'][:6])  - \
                       np.sum(state['leaving_vehicle'][:6])), 0)
        phase_2 = max((np.sum(state['coming_vehicle'][6:]) - \
                       np.sum(state['leaving_vehicle'][6:])), 0)


        if len(state['coming_vehicle'])!=12:
            phase_1 = max((np.sum(state['coming_vehicle'][3:6])+np.sum(state['coming_vehicle'][12:15]) - \
                      np.sum(state['leaving_vehicle'][3:6]) - np.sum(state['leaving_vehicle'][12:15])),0)
            phase_2 = max((np.sum(state['coming_vehicle'][21:24])+np.sum(state['coming_vehicle'][30:33]) - \
                      np.sum(state['leaving_vehicle'][21:24]) - np.sum(state['leaving_vehicle'][30:33])), 0)
            phase_3 = max((np.sum(state['coming_vehicle'][0:3]) + np.sum(state['coming_vehicle'][9:12])-\
                      np.sum(state['leaving_vehicle'][0:3]) - np.sum(state['leaving_vehicle'][9:12])), 0)
            phase_4 = max((np.sum(state['coming_vehicle'][18:21]) + np.sum(state['coming_vehicle'][27:30])- \
                      np.sum(state['leaving_vehicle'][18:21]) - np.sum(state['leaving_vehicle'][27:30])), 0)

            self.action = np.argmax([phase_1, phase_2, phase_3, phase_4])
            if state["cur_phase"][0] == self.action:
                self.current_phase_time += 1
            else:
                self.current_phase_time = 0
            return self.action


        if self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
            self.action = np.argmax([phase_1, phase_2])
            if state["cur_phase"][0] == self.action:
                self.current_phase_time += 1
            else:
                self.current_phase_time = 0
            return self.action

        else:
            if state["time_this_phase"][0] >= self.dic_agent_conf["FIXED_TIME"][cur_phase] and cur_phase != -1:
                self.current_phase_time = 0
                self.action = 1
                return 1
            else:
                self.current_phase_time += 1
                self.action = 0
                return 0

    def round_up(self, x, b=5):
        round_x = (b * np.ceil(x.astype(float) / b)).astype(int)
        round_x[np.where(round_x < self.dic_agent_conf["MIN_PHASE_TIME"])] = self.dic_agent_conf["MIN_PHASE_TIME"]
        return round_x

    def get_phase_split(self, traffic_demand, ratio):

        h = 2.45
        tL_set = 5
        tL = 14
        PHF = 1
        vc = 1
        N = 4
        vehicles_count_for_critical_lane_phase = traffic_demand * (1 + ratio)
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
            phase_split = np.copy(vehicles_count_for_critical_lane_phase) / np.sum(
                vehicles_count_for_critical_lane_phase) * effect_cycle_length
        else:
            phase_split = np.full(shape=(len(vehicles_count_for_critical_lane_phase),),
                                  fill_value=1 / len(vehicles_count_for_critical_lane_phase)) * effect_cycle_length

        phase_split = int(phase_split) + 1
        green = int(phase_split / (1 + ratio)) + 1
        red = int(phase_split / (1 + ratio) * ratio) + 1

        phase_split = np.array([green, red])


        phase_split = self.round_up(phase_split, b=self.dic_agent_conf["MIN_PHASE_TIME"])



        if self.IF_MULTI == True:
            green1 = green / 7
            green2 = green / 7 * 6
            red1 = red / 7
            red2 = red / 7 * 6
            phase_split = np.array([green2,green1,red2,red1])

            phase_split = self.round_up(phase_split, b=self.dic_agent_conf["MIN_PHASE_TIME"])


            return phase_split


        return phase_split

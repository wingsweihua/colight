import sys
sys.path.append("..")
from agent import Agent
import random


class FixedtimeAgent(Agent):


    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round,
                    ):

        super(FixedtimeAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path)

        self.current_phase_time = 0

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
        #print(state)
        # print(state["time_this_phase"][0], self.dic_agent_conf["FIXED_TIME"][cur_phase], cur_phase)

        if self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
            if state["time_this_phase"][0] >= self.dic_agent_conf["FIXED_TIME"][cur_phase] and cur_phase != -1:
                self.current_phase_time = 0
                self.action = (cur_phase + 1) % len(self.dic_traffic_env_conf["PHASE"])
                return (cur_phase + 1) % len(self.dic_traffic_env_conf["PHASE"])
            else:
                self.action = cur_phase
                self.current_phase_time += 1
                return cur_phase
        else:
            if state["time_this_phase"][0] >= self.dic_agent_conf["FIXED_TIME"][cur_phase] and cur_phase != -1:
                self.current_phase_time = 0
                self.action = 1
                return 1
            else:
                self.current_phase_time += 1
                self.action = 0
                return 0


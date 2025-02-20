import numpy as np
from Memory import BatchMemory
from Parameters import *


class LagrangeMethod:
    def __init__(self, lagrange_target_lane, lagrange_target_coll, lagrange_target_cros):
        self.lagrange_target_lane = lagrange_target_lane
        self.lagrange_target_coll = lagrange_target_coll
        self.lagrange_target_cros = lagrange_target_cros
        if BOOL_RCPO:
            self.lagrange_multiplier_lane = 0
            self.lagrange_multiplier_coll = 0
            self.lagrange_multiplier_cros = 0
        else:
            self.lagrange_multiplier_lane = LAGRANGE_FIXED_LANE
            self.lagrange_multiplier_coll = LAGRANGE_FIXED_COLL
            self.lagrange_multiplier_cros = LAGRANGE_FIXED_CROS

        self.lr_lane = LAGRANGE_LR_LANE
        self.lr_coll = LAGRANGE_LR_COLL
        self.lr_cros = LAGRANGE_LR_CROS
        self.gamma = 1

        # batch batch_memory
        self.driver_memory = BatchMemory('driver')

        # performance record
        self.J_cost_lane = 0
        self.J_cost_coll = 0
        self.J_cost_cros = 0
        self.record_cost_lane_buffer = []
        self.record_cost_coll_buffer = []
        self.record_cost_cros_buffer = []

    def lagrange_bound(self):
        self.lagrange_multiplier_lane = float(np.clip(self.lagrange_multiplier_lane, 0, 5000))
        self.lagrange_multiplier_coll = float(np.clip(self.lagrange_multiplier_coll, 0, 5000))
        self.lagrange_multiplier_cros = float(np.clip(self.lagrange_multiplier_cros, 0, 5000))
    
    def summerize(self, J_cost_lane, J_cost_coll, J_cost_cros):
        self.J_cost_lane = J_cost_lane
        self.J_cost_coll = J_cost_coll
        self.J_cost_cros = J_cost_cros    
        
    def lagrage_update(self):
        num_batch = len(self.driver_memory.batch_buffer)

        for batch_index in range(num_batch):
            worker_memory_buffer = self.driver_memory.batch_buffer[batch_index]

            # J_C: performance of cost
            discounted_cost_lane = 0
            discounted_cost_coll = 0
            discounted_cost_cros = 0
            for cost_lane, is_terminal in zip(reversed(worker_memory_buffer['cost_lane']),
                                              reversed(worker_memory_buffer['is_terminals'])):
                discounted_cost_lane = cost_lane + int(not is_terminal) * self.gamma * discounted_cost_lane
            for cost_coll, is_terminal in zip(reversed(worker_memory_buffer['cost_coll']),
                                              reversed(worker_memory_buffer['is_terminals'])):
                discounted_cost_coll = cost_coll + int(not is_terminal) * self.gamma * discounted_cost_coll
            for cost_cros, is_terminal in zip(reversed(worker_memory_buffer['cost_cros']),
                                              reversed(worker_memory_buffer['is_terminals'])):
                discounted_cost_cros = cost_cros + int(not is_terminal) * self.gamma * discounted_cost_cros
            J_cost_lane = discounted_cost_lane/len(worker_memory_buffer['cost_lane'])
            J_cost_coll = discounted_cost_coll
            J_cost_cros = discounted_cost_cros

            if BOOL_RCPO:
                self.lagrange_multiplier_lane += self.lr_lane * (J_cost_lane - self.lagrange_target_lane)
                self.lagrange_multiplier_coll += self.lr_coll * (J_cost_coll - self.lagrange_target_coll)
                self.lagrange_multiplier_cros += self.lr_cros * (J_cost_cros - self.lagrange_target_cros)
            else:
                self.lagrange_multiplier_lane = LAGRANGE_FIXED_LANE
                self.lagrange_multiplier_coll = LAGRANGE_FIXED_COLL
                self.lagrange_multiplier_cros = LAGRANGE_FIXED_CROS

            self.lagrange_bound()
            self.summerize(J_cost_lane, J_cost_coll, J_cost_cros)

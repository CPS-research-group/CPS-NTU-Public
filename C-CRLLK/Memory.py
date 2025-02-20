import copy
from Parameters import *

buffer = {'meta': [],
          'actions': [],
          'states': [],
          'log_probs': [],
          'rewards': [],
          'return': [],
          'is_terminals': [],
          'cost_lane': [],
          'cost_coll': [],
          'cost_cros': []}

env_recorder = {'total_step': 0,
                'total_reward': 0,
                'total_done': 0,
                'bool_learn': 0,
                'total_reward_hat': 0,
                'cost_lane': 0,
                'cost_coll': 0,
                'cost_cros': 0,
                'total_distance': 0,
                'total_angle': 0,
                'robot_speed': 0,
                'total_lane': 0}

loss_recorder = {'total_loss': 0,
                 'actor_loss': 0,
                 'critic_loss': 0,
                 'entropy_loss': 0}


# Memory for data saving in package
class SubMemory:
    def __init__(self):
        self.type = 'worker'
        self.buffer = buffer.copy()
        self.recorder = env_recorder.copy()

    def clear_memory(self):
        for key in self.buffer.keys():
            del self.buffer[key][:]
        for key in self.recorder.keys():
            self.recorder[key] = 0


class BatchMemory:
    def __init__(self, memory_type):
        """
        Conclude SubMemory in list.
        Note:
            - [dict{SubMemory.buffer}, dict{SubMemory.buffer}, ...] inside is buffer level.
            - never get recorder from buffer.
        """
        self.type = memory_type    # 'driver' or 'runner'
        self.batch_buffer = []
        self.recorder = env_recorder.copy()

    def clear_memory(self):
        # del self.batch_buffer[:]
        for sub_buffer in self.batch_buffer:
            for key in sub_buffer.keys():
                del sub_buffer[key][:]
            # del sub_buffer
        del self.batch_buffer[:]
        for key in self.recorder.keys():
            self.recorder[key] = 0
        
        # print('clear buffer', self.batch_buffer)

    def push_memory(self, input_memory):
        """
        Collect the memory in runner or driver level.
        :param input_memory: BatchMemory or SubMemory.
        :return:
        """
        if input_memory.type == 'worker':
            # we dont collect buffer with large variance
            if len(input_memory.buffer['actions']) > MIN_BUFFER_LENGTH:
                self.batch_buffer.append(input_memory.buffer)
        elif input_memory.type == 'runner':
            self.batch_buffer += input_memory.batch_buffer

        for key in self.recorder:
            self.recorder[key] += input_memory.recorder[key]
        # print('we are', self.type, 'input_memory.type', input_memory.type, self.recorder)

    def batch_drop(self):
        """
        In driver/runner level, run after push_memory(). This is the case where combine all
        the runner data into one buffer.
            - Remove the data we don't want.
        :return:
        """
        # for sub_buffer in self.batch_buffer:
        #     if len(sub_buffer['actions']) < MIN_BUFFER_LENGTH:
        #         print('delelet it')
        #         for key in sub_buffer.keys():
        #             del sub_buffer[key][:]
        #             del sub_buffer[key]
        #         del sub_buffer
        tmp_buffer = []
        for sub_buffer in self.batch_buffer:
            if len(sub_buffer['actions']) < MIN_BUFFER_LENGTH:
                print('delete sub-buffer')
                continue
            tmp_buffer.append(copy.deepcopy(sub_buffer))
            for key in sub_buffer.keys():
                del sub_buffer[key][:]
        for sub_buffer in self.batch_buffer:
            for key in sub_buffer.keys():
                del sub_buffer[key][:]
        del self.batch_buffer[:]
        self.batch_buffer = tmp_buffer

    def buffer_sort(self):
        # seeding: update network in order every time
        assert self.type == 'driver'
        self.batch_buffer = sorted(self.batch_buffer, key=lambda d: d['meta'][0])

    def convert_to_tensor(self):
        """
        Convert memory to training tensor. Extract data from buffer.
        :return:
        """
        raise NotImplementedError


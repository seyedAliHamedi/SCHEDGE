import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR


from configs import learning_config, environment_config
from environment.util import Utility


class Agent(mp.Process):
    def __init__(self, name, scheduler, barrier, shared_state):
        super(Agent, self).__init__()

        self.name = name
        self.devices = shared_state.db_devices
        self.util = Utility(devices=self.devices)
        self.scheduler = scheduler

        self.assigned_job = None  # current job assigned to the agent
        # flag to control agent's execution loop
        self.runner_flag = mp.Value('b', True)
        self.barrier = barrier  # barrier for synchronization across processes
        self.state = shared_state  # shared state between agents
        # counter for timeout detection
        self.time_out_counter = environment_config['time_out_counter']
        self.t_counter = 0  # tracks agent's wait time

    def init_logs(self):
        """Initialize logs for the agent."""
        self.state.agent_log[self.assigned_job] = {}
        self.reward_log, self.time_log, self.energy_log = [], [], []
        self.kind_fails_log = self.safe_fails_log = self.queue_fails_log = self.battery_fails_log = self.fails_log = 0
        self.iot_usuage = self.mec_usuage = self.cc_usuage = 0

    def run(self):
        """Main loop where the agent keeps running, processing jobs."""
        while self.runner_flag:
            self.barrier.wait()  # wait for all agents to be synchronized
            if self.assigned_job is None:
                self.assigned_job = self.state.assign_job_to_agent()
                if self.assigned_job is None:
                    continue
                # reset the status of the agent
                self.t_counter = 0
                self.init_logs()

            try:
                task_queue = self.state.preprocessor.get_agent_queue().get(self.assigned_job)
            except:
                continue

            if task_queue is None:
                continue

            self.t_counter += 1
            if self.t_counter >= self.time_out_counter and current_job and len(current_job["runningTasks"]) == 0 and len(current_job["remainingTasks"]) == 0:
                self._timeout_on_job()
                continue

            for task in task_queue:
                self.schedule(task)

            # Check if the current job is complete
            try:
                current_job = self.state.jobs[self.assigned_job]
            except:
                continue
            if current_job and len(current_job["runningTasks"]) + len(current_job["finishedTasks"]) == current_job["task_count"]:
                print(f"--  JOB {self.assigned_job} DONE")
                self.assigned_job = None

    def stop(self):
        self.runner_flag = False

    def schedule(self, current_task_id, gin=None, diversity=None, utilization=None):
        # retrieve the necessary data
        pe_state = self.state.PEs
        current_task = self.state.db_tasks[current_task_id]
        input_state = self.util.get_input(
            current_task)

        selected_device_index, selected_core_index, freq, vol = self.scheduler.schedule(
            task_features=input_state, task=current_task
        )
        # print(selected_device_index, selected_core_index, freq, vol)

        selected_device = self.devices[selected_device_index]
        reward, fail_flag, energy, time = self.state.apply_action(
            selected_device_index, selected_core_index, freq, vol, current_task_id)

        # saving agent logs
        self.update_agent_logs(reward, time, energy,
                               fail_flag, selected_device)

    def update_agent_logs(self, reward, time, energy, fail_flag, selected_device):
        """Update the logs for the agent based on task performance."""
        self.reward_log.append(reward)
        self.time_log.append(time)
        self.energy_log.append(energy)
        self.fails_log += sum(fail_flag)

        if selected_device['type'] == "iot":
            self.iot_usuage += 1
        elif selected_device['type'] == "mec":
            self.mec_usuage += 1
        elif selected_device['type'] == "cloud":
            self.cc_usuage += 1

        if fail_flag[0]:
            self.safe_fails_log += 1
        if fail_flag[1]:
            self.kind_fails_log += 1
        if fail_flag[2]:
            self.queue_fails_log += 1
        if fail_flag[3]:
            self.battery_fails_log += 1

    def save_agent_log(self, loss):
        """Save the logs of the agent after processing a job."""
        job_length = len(self.energy_log)
        result = {
            "loss": loss,
            "reward": sum(self.reward_log) / job_length,
            "time": sum(self.time_log) / job_length,
            "energy": sum(self.energy_log) / job_length,
            "safe_fails": self.safe_fails_log / job_length,
            "kind_fails": self.kind_fails_log / job_length,
            "queue_fails": self.queue_fails_log / job_length,
            "battery_fails": self.battery_fails_log / job_length,
            "fails": self.fails_log / job_length,
            "iot_usuage": self.iot_usuage / job_length,
            "mec_usuage": self.mec_usuage / job_length,
            "cc_usuage": self.cc_usuage / job_length,
        }
        self.state.save_agent_log(self.assigned_job, result, self.path_history)

    ####### UTILITY FUNCTIONS #######

    def _timeout_on_job(self):
        """Handle timeout when a job is taking too long."""
        print("JOB Stuck ", self.assigned_job)
        self.state.remove_job(self.assigned_job)
        self.assigned_job = None

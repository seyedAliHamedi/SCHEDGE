import numpy as np
import pandas as pd
import torch.multiprocessing as mp

from environment.util import *
from environment.pre_processing import Preprocessing
from environment.window_manager import WindowManager

from configs import environment_config


class State:
    def __init__(self, devices, jobs, tasks, manager):
        # OPTIMIZATION: Initialize task lookup dict once
        self.db_devices = devices
        self.db_jobs = jobs
        self.db_tasks = {item['id']: item for item in tasks}

        self.PEs = manager.list()
        self.jobs = manager.dict()
        self.util = Utility(devices=self.db_devices)
        self.init_PEs(self.db_devices, manager)

        self.task_window = manager.list()

        self.preprocessor = Preprocessing(state=self, manager=manager)
        self.window_manager = WindowManager(state=self, manager=manager)

        self.agent_log = manager.dict()
        self.paths = manager.list()
        self.display = environment_config['display']
        self.lock = mp.Lock()

    def init_PEs(self, PEs, manager):
        # OPTIMIZATION: Batch initialize PEs
        for pe in PEs:
            num_cores = pe["num_cores"]
            max_queue = pe["maxQueue"]

            queues = manager.list([
                manager.list([(0, -1)] * max_queue)
                for _ in range(num_cores)
            ])

            self.PEs.append({
                "type": pe["type"],
                "batteryLevel": 100.0,
                "queue": queues,
            })

    def set_jobs(self, jobs, manager):
        for job in jobs:
            self.jobs[job["id"]] = {
                "id": job["id"],
                "task_count": job["task_count"],
                "finishedTasks": manager.list([]),
                "runningTasks": manager.list([]),
                "remainingTasks": manager.list([]),
                "remainingDeadline": job["deadline"],
            }

    def remove_job(self, job_id):
        try:
            with self.lock:
                del self.jobs[job_id]
        except:
            return

    def apply_action(self, pe_ID, core_i, freq, volt, task_ID, utilization=None, diversity=None, gin=None):
        try:
            pe_dict = self.PEs[pe_ID]
            pe = self.db_devices[pe_ID]
            task = self.db_tasks[task_ID]
            if task["job_id"] not in self.jobs.keys():
                return 0, [0, 0, 0, 0], 0, 0
            job_dict = self.jobs[task["job_id"]]
        except:
            return self.apply_action(pe_ID, core_i, freq, volt, task_ID, utilization, diversity, gin)

        with self.lock:
            # OPTIMIZATION: Calculate totals before lock
            total_t, total_e = calc_total(
                pe, task, [self.db_tasks[pre_id] for pre_id in task["predecessors"]], core_i, 0)
            execution_time = min(total_t, 1)
            placing_slot = (execution_time, task_ID)

            queue_index, core_index, lag_time = self.find_place(
                pe_dict, core_i)

            fail_flags = [0, 0, 0, 0]
            if task["is_safe"] and not pe['is_safe']:
                fail_flags[0] = 1
            if task["task_kind"] not in pe["acceptable_tasks"]:
                fail_flags[1] = 1
            if queue_index == -1 and core_index == -1:
                fail_flags[2] = 1

            if sum(fail_flags) > 0:
                return sum(fail_flags) * reward_function(punish=True), fail_flags, 0, 0

            pe_dict["queue"][core_index] = pe_dict["queue"][core_index][:queue_index] + [placing_slot] + \
                pe_dict["queue"][core_index][queue_index + 1:]

            job_dict["runningTasks"].append(task_ID)
            try:
                job_dict["remainingTasks"].remove(task_ID)
            except:
                pass

            self.preprocessor.queue.remove(task_ID)

        battery_punish, batteryFail = self.util.checkBatteryDrain(
            total_e, pe_dict, pe)
        if batteryFail:
            fail_flags[3] = 1
            return sum(fail_flags) * reward_function(punish=True), fail_flags, 0, 0

        return reward_function(t=total_t + lag_time, e=total_e) + battery_punish, fail_flags, total_e, total_t + lag_time

    def save_agent_log(self, assigned_job, dict, path_history):
        with self.lock:
            self.agent_log[assigned_job] = dict
            self.paths.append(path_history)

    def assign_job_to_agent(self):
        with self.lock:
            return self.preprocessor.assign_job()

    def update(self, manager):
        self.window_manager.run()
        self.__update_jobs(manager)
        self.__update_PEs()
        self.preprocessor.run()

        if self.display:
            self.__display_state()

    def __update_jobs(self, manager):
        self.__add_new_active_jobs(self.task_window, manager)

        removing_items = []
        for job_id in list(self.jobs.keys()):
            job = self.jobs[job_id]
            # OPTIMIZATION: Update deadline in one operation
            self.jobs[job_id] = {
                **job,
                "remainingDeadline": job["remainingDeadline"] - 1
            }

            if len(job["finishedTasks"]) == job["task_count"]:
                removing_items.append(job['id'])

        for item in removing_items:
            del self.jobs[item]

    def __update_PEs(self):
        # Loop over all processing elements (PEs)
        for pe_index, pe_dict in enumerate(self.PEs):
            queue_list = pe_dict["queue"]

            # Prepare lists for updates
            updates = []
            finished_tasks = []

            # Collect updates and finished tasks in one pass
            for core_index, core_queue in enumerate(queue_list):
                first_task = core_queue[0]
                time_left, task_id = first_task

                # If the task is finished
                if time_left <= 0:
                    if task_id != -1:
                        finished_tasks.append(task_id)
                        # Append update: mark core as idle and reset task queue
                        updates.append(
                            (core_index, core_queue[1:] + [(0, -1)]))
                else:
                    # Task still running: reduce time left and keep core occupied
                    updates.append(
                        (core_index, [(time_left - 1, task_id)] + core_queue[1:]))

            # Process all finished tasks at once
            for task_id in finished_tasks:
                self.__task_finished(task_id)

            # Apply all updates at once to minimize manager.list() access
            for core_index, new_queue in updates:
                queue_list[core_index] = new_queue

    def find_place(self, pe, core_i):
        try:
            if pe['type'] == 'cloud' or True:
                for core_index, queue in enumerate(pe["queue"]):
                    if queue[0][1] == -1:
                        return 0, core_index, 0
            return -1, -1, -1
        except:
            return self.find_place(pe, core_i)

    def __add_new_active_jobs(self, new_tasks, manager):
        if self.display:
            print(f"new window {new_tasks}")

        for task_id in new_tasks:
            task = self.db_tasks[task_id]
            job = self.db_jobs[task['job_id']]

            if not self.__is_active_job(task['job_id']):
                self.set_jobs([job], manager)
            self.jobs[task['job_id']]["remainingTasks"].append(task_id)

    def __is_active_job(self, job_ID):
        return job_ID in self.jobs

    def __task_finished(self, task_ID):
        try:
            task = self.db_tasks[task_ID]
            job_ID = task["job_id"]
            job = self.jobs[job_ID]
            task_suc = task['successors']
        except:
            return

        if job is None or task_ID not in job["runningTasks"]:
            return

        for t in task_suc:
            self.db_tasks[t]['pred_count'] -= 1
            if self.db_tasks[t]['pred_count'] <= 0:
                if t in job['remainingTasks']:
                    self.preprocessor.queue.append(t)

        job["finishedTasks"].append(task_ID)
        job["runningTasks"].remove(task_ID)

    def __display_state(self):
        print("PEs::")
        pe_data = {}
        for pe_id, pe in enumerate(self.PEs):
            pe_data[pe_id] = {
                "queue": [list(core_queue) for core_queue in pe["queue"]]
            }
        print('\033[94m', pd.DataFrame(pe_data), '\033[0m', '\n')

        print("Jobs::")
        job_data = {}
        for job_id, job in self.jobs.items():
            job_data[job_id] = {
                "runningTasks": list(job["runningTasks"]),
            }
        print('\033[92m', pd.DataFrame(job_data), '\033[0m', "\n")

        print("|" * 89)

import numpy as np
import torch
import pandas as pd
import torch.multiprocessing as mp

from data.gen import Generator
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
        
        self.device_usage = manager.list([manager.list([1]) for _ in range(len(self.db_devices))])
        self.max_usage_history = 100

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
    
    def update_utilization_metrics(self):
        utilization = [sum(usage) for usage in self.device_usage ]
        used_devices_count = sum(1 for usage in self.device_usage  if 1 in usage)
        self.gin = gini_coefficient(utilization)
        self.diversity = used_devices_count / len(self.db_devices)
        self.utilization = torch.tensor(utilization, dtype=torch.float)
        
    def apply_action(self, pe_ID, core_i, freq, volt, task_ID, failed=False):
        try:
            pe_dict = self.PEs[pe_ID]
            pe = self.db_devices[pe_ID]
            task = self.db_tasks[task_ID]
            job_dict = self.jobs[task["job_id"]]
        except Exception as e:
            if failed:
                return 0, [0,0,0,0], 0, 0
            return self.apply_action(pe_ID, core_i, freq, volt, task_ID,True)

        with self.lock:
            # OPTIMIZATION: Calculate totals before lock
            total_t, total_e = calc_total(pe, task, [self.db_tasks[pre_id] for pre_id in task["predecessors"]], core_i, 0)
            execution_time = min(total_t, 1)
            placing_slot = (execution_time, task_ID)

            queue_index, core_index, lag_time = self.find_place(pe_dict, core_i)

            fail_flags = [0, 0, 0, 0]
            if task["is_safe"] and not pe['is_safe']:
                fail_flags[0] = learning_config['safe_punish'] and 1
            if task["task_kind"] not in pe["acceptable_tasks"]:
                fail_flags[1] = learning_config['kind_punish'] and 1
            if queue_index == -1 and core_index == -1:
                fail_flags[2] = learning_config['queue_punish'] and 1

            if sum(fail_flags) > 0 or (queue_index == -1 and core_index == -1):
                return sum(fail_flags) * reward_function(punish=True), fail_flags, 0, 0

            pe_dict["queue"][core_index] = pe_dict["queue"][core_index][:queue_index] + [placing_slot] + \
                                         pe_dict["queue"][core_index][queue_index + 1:]

            job_dict["runningTasks"].append(task_ID)
            try:
                job_dict["remainingTasks"].remove(task_ID)
                self.preprocessor.queue.remove(task_ID)
            except:
                pass
        try:
            self.update_device_usage(pe_ID)
        except:
            pass
        battery_punish = 0
        if learning_config['drain_battery']:
            battery_punish, batteryFail = self.util.checkBatteryDrain(total_e, pe_dict, pe)
            if batteryFail:
                fail_flags[3] = 1
                return sum(fail_flags) * reward_function(punish=True), fail_flags, 0, 0

        lambda_penalty = 0
        if learning_config['utilization']:
            lambda_diversity = learning_config["max_lambda"] * (1 - self.diversity)
            lambda_gini = learning_config["max_lambda"] * self.gin
            lambda_penalty = learning_config["alpha_diversity"] * lambda_diversity + learning_config["alpha_gin"] * lambda_gini

        reg_t = total_t
        reg_e=total_e
        if learning_config['regularize_output']:
            reg_t = self.util.regularize_output(total_t=total_t)
            reg_e = self.util.regularize_output(total_e=total_e)

        return reward_function(t=reg_t , e=reg_e) * (1 - lambda_penalty * self.utilization[pe_ID]) + battery_punish, fail_flags, total_e, total_t

    def save_agent_log(self, assigned_job, dict, path_history):
        with self.lock:
            self.agent_log[assigned_job] = dict
            self.paths.append(path_history)
            

    def assign_job_to_agent(self):
        with self.lock:
            return self.preprocessor.assign_job()

    def update_device_usage(self, device_index, used=True):
        """Track device usage with thread safety."""
        with self.lock:
            for d_index,usage in enumerate(self.device_usage):
                usage.append(1 if d_index==device_index else 0)
            if len(usage) > self.max_usage_history:
                usage[:] = usage[1:]

    def add_device(self, manager,new_device=None):
        """Add a new device to the system."""
        with self.lock:
            try:
                if new_device is None:
                    new_device = Generator.generate_random_device()
                
                # Add to device database
                # self.db_devices.append(new_device)
                
                # Initialize PE state for new device
                new_pe = {
                    "type": new_device["type"],
                    "batteryLevel": 100.0,
                    "queue": manager.list(
                        [manager.list([(0, -1) for _ in range(new_device["maxQueue"])])
                         for _ in range(new_device["num_cores"])])
                }
                self.PEs.append(new_pe)
                
                # Initialize usage tracking
                self.device_usage.append(manager.list([1]))
                
                # Update utility with new device list
                self.util = Utility(devices=self.db_devices)
                
                return True
            except Exception as e:
                print(f"Error adding device: {e}")
                return False

    def remove_device(self, device_index):
        """Remove a device from the system."""
        with self.lock:
            try:
                if device_index >= len(self.db_devices):
                    return False
                    
                device = self.db_devices[device_index]
                if device['type'] == 'cloud' :
                    return False
                
                # Remove from device database
                # del self.db_devices[device_index]
                
                # Remove PE state
                del self.PEs[device_index]
                
                # Remove usage tracking
                del self.device_usage[device_index]
                
                # Update utility with new device list
                self.util = Utility(devices=self.db_devices)
                
                return True
            except Exception as e:
                print(f"Error removing device: {e}")
                return False

    def update(self, manager):
        # Original update functionality
        self.window_manager.run()
        self.__update_jobs(manager)
        self.__update_PEs()
        self.preprocessor.run()

        # Display status if enabled
        if self.display:
            self.__display_status()
    
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
                        updates.append((core_index, core_queue[1:] + [(0, -1)]))
                else:
                    # Task still running: reduce time left and keep core occupied
                    updates.append((core_index, [(time_left - 1, task_id)] + core_queue[1:]))

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
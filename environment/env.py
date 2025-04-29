import os
import time
import traceback
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from environment.state import State
from environment.util import *
from data.gen import Generator

from model.agent import Agent
from model.utils import SharedAdam
from model.actor_critic import ActorCritic

import torch.multiprocessing as mp

from configs import environment_config,learning_config,devices_config


class Environment:

    def __init__(self):
        print("\n ---------------------- ")
        self.initialize()
        # the shared state and the manager for shared memory variables/dictionaries/lists
        manager = mp.Manager()
        self.manager = manager
        
        
        self.state = State(devices=self.devices,jobs=self.jobs,tasks=self.tasks,manager=manager)
        # the 3 global enteties of the shared state (preProcessor,windowManager,database)
        self.preprocessor = self.state.preprocessor
        self.window_manager = self.state.window_manager
        self.num_device_added=0
        self.num_device_removed=0
        self.display = environment_config['display']
        self.time_log = []
        print("Envionment initialized ")
        
        
    def initialize(self):
        print("initialize Envionment ....")
        self.devices = Generator.get_devices()
        self.jobs = Generator.get_jobs()
        self.tasks = Generator.get_tasks()
        print("Data loaded")
        
    def run(self):
        # define the global Actor-Critic and the shared optimizer (A3C)
        self.global_actor_critic = ActorCritic(devices=self.devices)
        self.global_actor_critic.share_memory()
        global_optimizer = SharedAdam(self.global_actor_critic.parameters())
        # setting up workers and their barriers
        self.workers = []
        self.barrier = mp.Barrier(environment_config['multi_agent'] + 1)
        # kick off the state
        self.state.update(self.manager)
        self.state.update_utilization_metrics()
        print("Starting agents .....")
        for i in range(environment_config['multi_agent']):
            # organize and start the agents
            worker = Agent(name=f'agent_{i}', global_actor_critic=self.global_actor_critic,
                           global_optimizer=global_optimizer, barrier=self.barrier, shared_state=self.state,
                           )
            self.workers.append(worker)
            worker.start()

        iteration = 0
        try:
            print("Simulation starting...")
            while True:
                if learning_config['scalability']:
                    a = np.random.random()
                    b = np.random.random()
                    if a < learning_config['add_device_iterations'] and self.num_device_added-self.num_device_removed <=2:
                       self.num_device_added +=1
                       self.add_device()
                    if b < learning_config['remove_device_iterations'] and self.num_device_removed-self.num_device_added <=2:
                       self.num_device_removed +=1
                       self.remove_device()

                       
                if iteration % 10 == 0 and iteration != 0:
                    print(f"iteration : {iteration}", len(self.state.jobs))
                    if learning_config['utilization']:
                        self.state.update_utilization_metrics()
                    if iteration % 100 == 0:
                        self.save_time_log()
                        self.make_agents_plots()
                        self.save_results()
                        self.save_model()
                        self.load_model()
                    
                starting_time = time.time()
                self.check_dead_iot_devices()
                self.state.update(self.manager)
                self.barrier.wait()
                
                
                iteration += 1
                time_len = time.time() - starting_time
                self.time_log.append(time_len)

        except Exception as e:
            print("Caught an unexpected exception:", e)
            traceback.print_exc()
        finally:
            print("Simulation Finished")
            print("Saving Logs......")
            self.save_time_log(learning_config['result_time_plot'])
            self.make_agents_plots()
            self.save_results()

            # stopping and terminating the workers
            for worker in self.workers:
                worker.stop()
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

    def add_device(self):
        print("added a device")
        """Add device and propagate changes downstream."""
        new_device = Generator.generate_random_device()
        self.devices.append(new_device)
        
        # Update state
        self.state.add_device(manager=self.manager,new_device=new_device)
        
        # Update global actor critic first
        self.global_actor_critic.add_device(new_device)
        # Update all workers
        for worker in self.workers:
            worker.local_actor_critic.add_device(new_device)
            worker.add_device(new_device)
            
        # Wait for all processes to finish updating
            
    def remove_device(self, device_index=None):
        print("device removed")
        """Remove device with synchronized update."""
        if len(self.devices) <= 1:
            return
        
        if device_index is None:
            adjusted_usage = self.state.device_usage
            if len(self.state.device_usage) < len(self.devices):
                adjusted_usage = self.state.device_usage + [[0]] * (len(self.devices) - len(self.state.device_usage))
            elif len(self.state.device_usage) > len(self.devices):
                adjusted_usage = self.state.device_usage[:len(self.devices)]
                
            # Calculate weights as inverse usage, avoid division by zero
            usage_sums = [sum(usage) for usage in adjusted_usage]
            weights = [1 / (usage_sum + 1) for usage_sum in usage_sums]  # Adding 1 to avoid zero division
            weights = np.array(weights) / np.sum(weights)  # Normalize to sum to 1

            # Select device index based on weights
            device_index = np.random.choice(range(len(self.devices)), p=weights)

            
        if self.devices[device_index]['type'] == 'cloud':
            return
            
        del self.devices[device_index]
        
        # Update state
        self.state.remove_device(device_index)
        
        # Update global actor critic first
        self.global_actor_critic.remove_device(device_index)
        
        # Then update all workers synchronously
        for worker in self.workers:
            worker.local_actor_critic.remove_device(device_index)
            worker.remove_device(device_index)
            
    def check_dead_iot_devices(self):
        """Remove IoT devices with depleted batteries."""
        removing_devices = []
        for idx, device in enumerate(self.devices):
            if (device['type'] == 'iot' and 
                self.state.PEs[idx]['batteryLevel'] < device['ISL'] * 100):
                removing_devices.append(idx)
        
        # Remove devices in reverse order
        for idx in sorted(removing_devices, reverse=True):
            self.remove_device(idx)
            print("IoT device removed due to depleted battery")

    def save_time_log(self, path=learning_config['result_time_plot']):
        if len(self.time_log) <= 10:
            return
        # saving time log gathered in the simulation
        y_values = self.time_log[10:]
        time_summary_log = {
            'total_time': sum(y_values),
            'average_time': sum(y_values) / len(y_values),
            'max_time': max(y_values),
            'epochs': len(y_values),
        }
        plt.figure(figsize=(10, 5))
        plt.plot(y_values, marker='o', linestyle='-')
        plt.title("Sleeping time on each iteration")
        plt.xlabel("iteration")
        plt.ylabel("sleeping time")
        plt.grid(True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)

    def make_agents_plots(self, plot_path=learning_config['result_plot_path']):
        # Get filtered data
        filtered_data = {k: v for k, v in self.state.agent_log.items() if v}
        time_list = [v["time"] for v in filtered_data.values()]
        energy_list = [v["energy"] for v in filtered_data.values()]
        reward_list = [v["reward"] for v in filtered_data.values()]
        loss_list = [v["loss"] for v in filtered_data.values()]

        fails_list = [v["fails"] for v in filtered_data.values()]
        safe_fails_list = [v["safe_fails"] for v in filtered_data.values()]
        kind_fails_list = [v["kind_fails"] for v in filtered_data.values()]
        queue_fails_list = [v["queue_fails"] for v in filtered_data.values()]
        battery_fails_list = [v["battery_fails"] for v in filtered_data.values()]
        iot_usage = [v["iot_usuage"] for v in filtered_data.values()]
        mec_usuage = [v["mec_usuage"] for v in filtered_data.values()]
        cc_usuage = [v["cc_usuage"] for v in filtered_data.values()]

        path_history = self.state.paths

        fig = plt.figure(figsize=(25, 40))
        gs = fig.add_gridspec(5, 2)  # 6 rows,  columns

        # Plot Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(loss_list, label='Loss', color="blue", marker='o')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot Time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_list, label='Time', color="red", marker='o')
        ax2.set_title('Time')
        ax2.legend()
        ax2.grid(True)

        # Plot Energy
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(energy_list, label='Energy', color="green", marker='o')
        ax3.set_title('Energy')
        ax3.legend()
        ax3.grid(True)

        # Plot All Fails
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(fails_list, label='ALL Fails', color="purple", marker='o')
        ax4.set_title('Fail')
        ax4.legend()
        ax4.grid(True)

        # Plot Safe Fails
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(safe_fails_list, label='Safe task Fail', color="purple", marker='o')
        ax5.set_title('Fail')
        ax5.legend()
        ax5.grid(True)

        # Plot Kind Fails
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(kind_fails_list, label='Task kind Fail', color="purple", marker='o')
        ax6.set_title('Fail')
        ax6.legend()
        ax6.grid(True)

        # Plot Device Type Usage
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(iot_usage, label='IoT Usage', color='blue', marker='o')
        ax7.plot(mec_usuage, label='MEC Usage', color='orange', marker='x')
        ax7.plot(cc_usuage, label='Cloud Usage', color='green', marker='s')
        ax7.set_title('Devices Usage History')
        ax7.set_xlabel('Epochs')
        ax7.set_ylabel('Usage')
        ax7.legend()
        ax7.grid(True)

        # Plot Path History Heatmap
        if path_history and len(path_history) > 0:
            ax8 = fig.add_subplot(gs[3, 1])
            output_classes = make_paths(len(path_history[0][0]))
            path_counts = np.zeros((len(path_history), len(output_classes)))

            for epoch in range(len(path_history)):
                epoch_paths = path_history[epoch]
                for path in epoch_paths:
                    path_index = output_classes.index(path)
                    path_counts[epoch, path_index] += 1

            # Find threshold value
            flat_counts = path_counts.flatten()
            sorted_counts = np.sort(flat_counts)
            if len(sorted_counts) > 10:
                threshold = sorted_counts[-11]
                path_counts = np.minimum(path_counts, threshold)

            sns.heatmap(path_counts, cmap="YlGnBu", xticklabels=output_classes, ax=ax8)
            ax8.set_title('Path History Heatmap')
            ax8.set_xlabel('Output Classes')
            ax8.set_ylabel('Epochs')

        ax9 = fig.add_subplot(gs[4:, :])  # Use last two rows and span all columns
        colors = ['blue'] * devices_config['iot']['num_devices'] + \
                ['orange'] * devices_config['mec']['num_devices'] + \
                ['green'] * devices_config['cloud']['num_devices']
        
        # Calculate device usage from state's device_usage list
        device_usage = [sum(usage) for usage in self.state.device_usage]
        ax9.bar(range(1, len(device_usage) + 1), device_usage, color=colors)
        ax9.set_title('PE Activity History')
        ax9.set_xlabel('Device')
        ax9.set_yticks([])
        ax9.set_xticks(range(1, len(device_usage) + 1))

        plt.tight_layout()
        plt.savefig(plot_path)
        
    def save_model(self, path=learning_config['checkpoint_file_path']):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.global_actor_critic.state_dict(), path)
        print(f"Model saved at {path}")
        
    def load_model(self, path=learning_config['checkpoint_file_path']):
        if os.path.exists(path):
            self.global_actor_critic.load_state_dict(torch.load(path,weights_only=True))
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")


        
        
    def save_results(self):
        filtered_data = {k: v for k, v in self.state.agent_log.items() if v}
        time_list = [v["time"] for v in filtered_data.values()]
        energy_list = [v["energy"] for v in filtered_data.values()]
        reward_list = [v["reward"] for v in filtered_data.values()]
        loss_list = [v["loss"] for v in filtered_data.values()]

        fails_list = [v["fails"] for v in filtered_data.values()]
        safe_fails_list = [v["safe_fails"] for v in filtered_data.values()]
        kind_fails_list = [v["kind_fails"] for v in filtered_data.values()]
        queue_fails_list = [v["queue_fails"] for v in filtered_data.values()]
        battery_fails_list = [v["battery_fails"] for v in filtered_data.values()]
        iot_usage = [v["iot_usuage"] for v in filtered_data.values()]
        mec_usuage = [v["mec_usuage"] for v in filtered_data.values()]
        cc_usuage = [v["cc_usuage"] for v in filtered_data.values()]
        num_epoch = len(time_list)
        half_num_epoch = num_epoch // 2

        new_epoch_data = {
            "Setup": learning_config['rewardSetup'],
            "Punishment": learning_config['init_punish'],
            "diversity": self.state.diversity,
            "giniImact": self.state.gin,

            "Average Loss": sum(loss_list) / num_epoch,
            "Last Epoch Loss": loss_list[-1],
            
            "Battery Fail Percentage": np.count_nonzero(battery_fails_list) / len(battery_fails_list),
            "Task Fail Percentage": np.count_nonzero(kind_fails_list) / len(kind_fails_list),
            "Safe Fail Percentage": np.count_nonzero(safe_fails_list) / len(safe_fails_list),

            "Average Time": sum(time_list) / num_epoch,
            "Last Epoch Time": time_list[-1],

            "Average Energy": sum(energy_list) / num_epoch,
            "Last Epoch Energy": energy_list[-1],

            "Average Reward": sum(reward_list) / num_epoch,
            "Last Epoch Reward": reward_list[-1],

            "First 10 Avg Time": np.mean(time_list[:10]),
            "Mid 10 Avg Time": np.mean(time_list[half_num_epoch:half_num_epoch + 10]),
            "Last 10 Avg Time": np.mean(time_list[:-10]),

            "First 10 Avg Energy": np.mean(energy_list[:10]),
            "Mid 10 Avg Energy": np.mean(energy_list[half_num_epoch:half_num_epoch + 10]),
            "Last 10 Avg Energy": np.mean(energy_list[:-10]),

            "First 10 Avg Reward": np.mean(reward_list[:10]),
            "Mid 10 Avg Reward": np.mean(reward_list[half_num_epoch:half_num_epoch + 10]),
            "Last 10 Avg Reward": np.mean(reward_list[:-10]),

            "First 10 Avg Loss": np.mean(loss_list[:10]),
            "Mid 10 Avg Loss": np.mean(loss_list[half_num_epoch:half_num_epoch + 10]),
            "Last 10 Avg Loss": np.mean(loss_list[:-10]),

            "First 10 (total, task, safe) Fail": sum(fails_list[:10])/len(fails_list[:10]),
            "Mid 10 (total, task, safe) Fail": sum(fails_list[half_num_epoch:half_num_epoch + 10])/len(fails_list[half_num_epoch:half_num_epoch + 10]), 
            "Last 10 (total, task, safe) Fail": sum(fails_list[:-10])/len(fails_list[:-10]),
        }
        new_epoch_data_list = [new_epoch_data]

        df = None
        if os.path.exists(learning_config['result_summery_path']):
            df = pd.read_csv(learning_config['result_summery_path'])
            new_df = pd.DataFrame(new_epoch_data_list)
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(new_epoch_data_list)

        df.to_csv(learning_config['result_summery_path'], index=False)

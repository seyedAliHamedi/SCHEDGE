import os
import time
import traceback

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from environment.state import State
from environment.util import *
from data.gen import Generator

from model.agent import Agent
from model.schedulers import (
    OfflineScheduler,
    OnlineScheduler,
    DRLScheduler,
    HeuristicScheduler,
    EvolutionaryScheduler,
)


import torch.multiprocessing as mp

from configs import environment_config, learning_config, devices_config


class Environment:

    def __init__(self):
        print("\n ---------------------- ")
        self.initialize()
        # the shared state and the manager for shared memory variables/dictionaries/lists
        manager = mp.Manager()
        self.manager = manager

        self.state = State(devices=self.devices, jobs=self.jobs,
                           tasks=self.tasks, manager=manager)
        # the 3 global enteties of the shared state (preProcessor,windowManager,database)
        self.preprocessor = self.state.preprocessor
        self.window_manager = self.state.window_manager

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
        # setting up workers and their barriers
        self.workers = []
        barrier = mp.Barrier(environment_config['multi_agent'] + 1)
        # kick off the state
        self.state.update(self.manager)
        print("Starting agents .....")

        TEST_SCHEDULER = environment_config['scheduler_type']

        if TEST_SCHEDULER == "offline":
            scheduler = OfflineScheduler(self.state, self.devices)
        elif TEST_SCHEDULER == "online":
            scheduler = OnlineScheduler(self.state, self.devices)
        elif TEST_SCHEDULER == "drl":
            scheduler = DRLScheduler(
                self.state, self.devices, model=None)  # use your model
        elif TEST_SCHEDULER == "heuristic":
            scheduler = HeuristicScheduler(self.state, self.devices)
        elif TEST_SCHEDULER == "evolutionary":
            scheduler = EvolutionaryScheduler(self.state, self.devices)

        for i in range(environment_config['multi_agent']):

            worker = Agent(
                name=f'agent_{i}', scheduler=scheduler, barrier=barrier, shared_state=self.state)
            self.workers.append(worker)
            worker.start()

        iteration = 0
        try:
            print("Simulation starting...")
            while True:
                if iteration % 10 == 0 and iteration != 0:
                    print(f"iteration : {iteration}", len(self.state.jobs))
                    if iteration % 100 == 0:
                        self.save_time_log()
                        self.make_agents_plots()
                starting_time = time.time()
                self.state.update(self.manager)
                barrier.wait()
                # print("ITERATION " ,iteration,time.time()-starting_time)
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

            # stopping and terminating the workers
            for worker in self.workers:
                worker.stop()
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

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
        # saving the agent logs gatherd from the state
        filtered_data = {k: v for k, v in self.state.agent_log.items() if v}
        time_list = [v["time"] for v in filtered_data.values()]
        energy_list = [v["energy"] for v in filtered_data.values()]
        reward_list = [v["reward"] for v in filtered_data.values()]
        loss_list = [v["loss"] for v in filtered_data.values()]

        fails_list = [v["fails"] for v in filtered_data.values()]
        safe_fails_list = [v["safe_fails"] for v in filtered_data.values()]
        kind_fails_list = [v["kind_fails"] for v in filtered_data.values()]
        queue_fails_list = [v["queue_fails"] for v in filtered_data.values()]
        battery_fails_list = [v["battery_fails"]
                              for v in filtered_data.values()]
        iot_usage = [v["iot_usuage"] for v in filtered_data.values()]
        mec_usuage = [v["mec_usuage"] for v in filtered_data.values()]
        cc_usuage = [v["cc_usuage"] for v in filtered_data.values()]

        path_history = self.state.paths

        fig, axs = plt.subplots(6, 2, figsize=(15, 36))
        axs[0, 0].plot(loss_list,
                       label='Loss', color="blue", marker='o')
        axs[0, 0].set_title('Loss')
        axs[0, 0].legend()

        # Plot for time
        axs[0, 1].plot(time_list,
                       label='Time', color="red", marker='o')
        axs[0, 1].set_title('Time')
        axs[0, 1].legend()

        # Plot for energy
        axs[1, 0].plot(energy_list,
                       label='Energy', color="green", marker='o')
        axs[1, 0].set_title('Energy')
        axs[1, 0].legend()

        # Plot for all fail
        axs[1, 1].plot(fails_list,
                       label='ALL Fails', color="purple", marker='o')
        axs[1, 1].set_title('Fail')
        axs[1, 1].legend()

        axs[2, 0].plot(safe_fails_list,
                       label='Safe task Fail', color="purple", marker='o')
        axs[2, 0].set_title('Fail')
        axs[2, 0].legend()

        # Plot for kind fail
        axs[2, 1].plot(kind_fails_list,
                       label='Task kind Fail', color="purple", marker='o')
        axs[2, 1].set_title('Fail')
        axs[2, 1].legend()

        # Plot for queue fail
        axs[3, 0].plot(queue_fails_list,
                       label='Queue full Fail', color="purple", marker='o')
        axs[3, 0].set_title('Fail')
        axs[3, 0].legend()

        # Plot for battery fail
        axs[3, 1].plot(battery_fails_list,
                       label='Battery punish', color="purple", marker='o')
        axs[3, 1].set_title('Fail')
        axs[3, 1].legend()

        axs[4, 0].plot(iot_usage, label='IoT Usage', color='blue', marker='o')
        axs[4, 0].plot(mec_usuage, label='MEC Usage',
                       color='orange', marker='x')
        axs[4, 0].plot(cc_usuage, label='Cloud Usage',
                       color='green', marker='s')
        axs[4, 0].set_title('Devices Usage History')
        axs[4, 0].set_xlabel('Epochs')
        axs[4, 0].set_ylabel('Usage')
        axs[4, 0].legend()
        axs[4, 0].grid(True)

        # Heatmap for path history
        # print(path_history)
        if path_history and len(path_history) > 0:

            output_classes = make_paths(len(path_history[0][0]))
            path_counts = np.zeros((len(path_history), len(output_classes)))

            for epoch in range(len(path_history)):
                epoch_paths = path_history[epoch]

                for path in epoch_paths:
                    path_index = output_classes.index(path)
                    path_counts[epoch, path_index] += 1

            sns.heatmap(path_counts, cmap="YlGnBu",
                        xticklabels=output_classes, ax=axs[4, 1])
            axs[4, 1].set_title(f'Path History Heatmap ')
            axs[4, 1].set_xlabel('Output Classes')
            axs[4, 1].set_ylabel('Epochs')

        # colors = ['blue'] *  devices_config['iot']['num_devices'] + ['orange'] *  devices_config['mec']['num_devices'] + ['green'] * devices_config['cloud']['num_devices']
        # axs[5, 0].bar(range(1, len(self.state.device_usuages) + 1), [sum(x) for x in self.state.device_usuages], color=colors)
        # axs[5, 0].set_title('PE ACTIVITY History')
        # axs[5, 0].set_xlabel('Device')
        # axs[5, 0].set_yticks([])
        # axs[5, 0].set_xticks(range(1, len(self.state.device_usuages) + 1))
        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plots to an image file
        plt.savefig(plot_path)

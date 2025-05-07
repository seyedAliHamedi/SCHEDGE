from environment.util import calc_total
import random
import torch
import torch.nn as nn
from environment.util import calc_total  # Make sure this is imported


class BaseScheduler:
    def __init__(self, state, devices):
        self.state = state
        self.devices = devices

    def schedule(self, task_features=None, task=None):
        return 0, 0, 0, 0

    def update(self):
        pass


class OfflineScheduler(BaseScheduler):
    def schedule(self, task_features=None, task=None):
        for d_idx, device in enumerate(self.devices):
            if task["is_safe"] == 1 and device["is_safe"] != 1:
                continue
            if task["task_kind"] not in device["acceptable_tasks"]:
                continue
            for c_idx, core in enumerate(device["voltages_frequencies"]):
                freq, volt = max(core, key=lambda x: x[0])
                return d_idx, c_idx, freq, volt

        freq, volt = max(
            self.devices[0]["voltages_frequencies"][0], key=lambda x: x[0])
        return 0, 0, freq, volt


class OnlineScheduler(BaseScheduler):
    def __init__(self, state, devices, input_size=8, hidden_size=64):
        super().__init__(state, devices)
        self.total_cores = sum(
            len(dev["voltages_frequencies"]) for dev in devices
        )

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.total_cores)
        )

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    def schedule(self, task_features=None, task=None):
        x = torch.tensor(task_features, dtype=torch.float32).unsqueeze(0)
        scores = self.net(x)
        action = torch.argmax(scores, dim=1).item()

        count = 0
        for d_idx, device in enumerate(self.devices):
            for c_idx in range(len(device["voltages_frequencies"])):
                if count == action:
                    return d_idx, c_idx, 0, 0  # ← only return device, core
                count += 1

        return 0, 0, 0, 0  # fallback

    def update(self, loss):
        pass


class DRLScheduler(BaseScheduler):
    def __init__(self, state, devices, model):
        super().__init__(state, devices)
        self.model = model

    def schedule(self, task_features=None, task=None):
        action = self.model.predict_action(task_features)
        d, c, freq_idx = action
        freq, volt = self.devices[d]["voltages_frequencies"][c][freq_idx]
        return d, c, freq, volt

    def update(self):
        self.model.update()


class HeuristicScheduler(BaseScheduler):
    def schedule(self, task_features=None, task=None):
        best_score = float("inf")
        best = (0, 0)
        for d_idx, device in enumerate(self.devices):
            for c_idx, core_options in enumerate(device["voltages_frequencies"]):
                freq, volt = max(core_options, key=lambda x: x[0])
                dvfs_index = device["voltages_frequencies"][c_idx].index(
                    (freq, volt))

                total_time, total_energy = calc_total(device, task, [
                    self.state.db_tasks[pre] for pre in task["predecessors"]], c_idx, dvfs_index)
                queue_len = len(self.state.PEs[d_idx]["queue"][c_idx])
                penalty = 10 if (
                    task["is_safe"] and not device['is_safe']) else 0
                penalty += 10 if (task["task_kind"]
                                  not in device["acceptable_tasks"]) else 0
                score = total_time + total_energy + penalty
                if score < best_score:
                    best_score = score
                    best = (d_idx, c_idx)
        d, c = best
        freq, volt = max(
            self.devices[d]["voltages_frequencies"][c], key=lambda x: x[0])
        return d, c, freq, volt


class EvolutionaryScheduler(BaseScheduler):
    def schedule(self, task_features=None, task=None):
        population_size = 20
        generations = 5
        mutation_rate = 0.2

        task_pres = [self.state.db_tasks[pre] for pre in task["predecessors"]]

        def fitness(d, c, f_idx):
            try:
                device = self.devices[d]
                if c >= len(device["voltages_frequencies"]):
                    return float("-inf")
                if f_idx >= len(device["voltages_frequencies"][c]):
                    return float("-inf")

                freq, volt = device["voltages_frequencies"][c][f_idx]

                # Safety and task kind penalties
                penalty = 0
                if task["is_safe"] and not device["is_safe"]:
                    penalty += 10
                if task["task_kind"] not in device["acceptable_tasks"]:
                    penalty += 10

                total_time, total_energy = calc_total(
                    device, task, task_pres, c, f_idx)
                return -(total_time + total_energy + penalty)

            except Exception:
                return float("-inf")  # if any index error or data bug

        num_devices = len(self.devices)
        population = []
        for _ in range(population_size):
            d = random.randint(0, num_devices - 1)
            c = random.randint(
                0, len(self.devices[d]["voltages_frequencies"]) - 1)
            f_idx = random.randint(
                0, len(self.devices[d]["voltages_frequencies"][c]) - 1)
            population.append((d, c, f_idx))

        for _ in range(generations):
            scores = [fitness(d, c, f_idx) for d, c, f_idx in population]
            sorted_pop = [x for _, x in sorted(
                zip(scores, population), reverse=True)]
            top_half = sorted_pop[:population_size // 2]

            children = []
            while len(children) < population_size // 2:
                p1, p2 = random.sample(top_half, 2)
                d = random.choice([p1[0], p2[0]])
                c = random.choice([p1[1], p2[1]])
                f = random.choice([p1[2], p2[2]])

                if random.random() < mutation_rate:
                    d = random.randint(0, num_devices - 1)
                    c = random.randint(
                        0, len(self.devices[d]["voltages_frequencies"]) - 1)
                    f = random.randint(
                        0, len(self.devices[d]["voltages_frequencies"][c]) - 1)

                # safe add
                children.append((d, c, f))

            population = top_half + children

        best = max(population, key=lambda x: fitness(*x))
        d, c, f_idx = best
        freq, volt = self.devices[d]["voltages_frequencies"][c][f_idx]
        return d, c, freq, volt

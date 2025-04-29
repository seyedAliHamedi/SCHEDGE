import os
import ast
import random
import numpy as np
import pandas as pd
import networkx as nx

from configs import devices_config, jobs_config

class Generator:
    
    # File paths for device, job, and task data
    devices_path = os.path.join(os.path.dirname(__file__), "resources", "scattered_devices.csv")
    job_path = os.path.join(os.path.dirname(__file__), "resources", "jobs.csv")
    tasks_path = os.path.join(os.path.dirname(__file__), "resources", "tasks.csv")

    @classmethod
    def get_devices(cls):
        return cls._load_csv(cls.devices_path, cls.generate_devices)

    @classmethod
    def get_jobs(cls):
        return cls._load_csv(cls.job_path, cls.generate_jobs)

    @classmethod
    def get_tasks(cls):
        tasks = cls._load_csv(cls.tasks_path, cls.generate_jobs)
        for task in tasks:
            task['pred_count'] = len(task['predecessors'])
            task['live_state'] = {}
            task['live_state']["chosen_device_type"]= None
            task['live_state']["iot_predecessors"]= 0
            task['live_state']["mec_predecessors"]= 0
            task['live_state']["cloud_predecessors"]= 0
        return tasks 

    @classmethod
    def generate_jobs(cls):
        config = jobs_config
        max_deadline = config["max_deadline"]
        task_config = config["task"]
        tasks_data, jobs_data = [], []
        start_node_number = 1

        for job_id in range(config["num_jobs"]):
            if job_id % (config["num_jobs"] // 10) == 0:
                print(f"Generating jobs: {job_id / config['num_jobs'] * 100:.2f}%")

            num_nodes = random.randint(config["min_num_nodes_dag"], config["max_num_nodes_dag"])
            random_dag = cls.generate_random_dag(num_nodes)

            mapping = {i: i + start_node_number - 1 for i in range(1, num_nodes + 1)}
            random_dag = nx.relabel_nodes(random_dag, mapping)

            for node in random_dag.nodes:
                parents = list(random_dag.predecessors(node))
                children = list(random_dag.successors(node))
                task_info = cls._generate_task_info(node, parents, children, job_id, task_config)
                tasks_data.append(task_info)

            start_node_number += num_nodes

            job_info = {
                "id": job_id,
                "task_count": len(random_dag.nodes),
                "tasks_ID": [task["id"] for task in tasks_data if task["job_id"] == job_id],
                "deadline": np.random.randint(max_deadline-5, max_deadline+5),
            }
            jobs_data.append(job_info)

        pd.DataFrame(jobs_data).to_csv(cls.job_path, index=False)
        pd.DataFrame(tasks_data).to_csv(cls.tasks_path, index=False)
        return jobs_data, tasks_data

    @classmethod
    def generate_devices(cls):
        devices_data = []
        for device_type in ("iot", "mec", "cloud"):
            config = devices_config[device_type]
            for _ in range(config["num_devices"]):
                cpu_cores = config['num_cores'] if device_type == 'cloud' else int(
                    np.random.choice(config["num_cores"]))
                device_info = cls._generate_device_info(device_type, cpu_cores, config)
                devices_data.append(device_info)

        devices = pd.DataFrame(devices_data)
        os.makedirs(os.path.dirname(cls.devices_path), exist_ok=True)
        devices.to_csv(cls.devices_path, index=False)
        return devices

    @classmethod
    def _generate_device_info(cls, device_type, cpu_cores, config):
        return {
            "type": device_type,
            "num_cores": cpu_cores,
            "voltages_frequencies": [[config["voltage_frequencies"][i]
                                      for i in
                                      np.random.choice(len(config['voltage_frequencies']), size=3, replace=False)]
                                     for _ in range(cpu_cores)
                                     ],
            "ISL": cls._generate_value(config["isl"]),
            "capacitance": np.random.uniform(*config["capacitance"]) * 1e-9,
            "powerIdle": float(np.random.choice(config["powerIdle"])) * 1e-6,
            "battery_capacity": cls._generate_value(config["battery_capacity"]) * 1e6,
            "acceptable_tasks": list(np.random.choice(jobs_config["task"]["task_kinds"],
                                                      size=np.random.randint(config["num_acceptable_task"][0],
                                                                             config["num_acceptable_task"][1]),
                                                      replace=False)),
            "is_safe": int(np.random.choice([0, 1], p=[config["safe"][0], config["safe"][1]])),
            "maxQueue":config['maxQueue'],
        }

    @classmethod
    def _generate_value(cls, value_config):
        return -1 if value_config == -1 else np.random.uniform(value_config[0], value_config[1])

    @classmethod
    def _generate_task_info(cls, id, parents, children,  job_id, task_config):
        return {
            "id": id,
            "job_id": job_id,
            "predecessors": parents,
            "successors": children,
            "computational_load": np.random.randint(*task_config["computational_load"]) * 1e6,
            "input_size": np.random.randint(*task_config["input_size"]) * 1e6,
            "output_size": np.random.randint(*task_config["output_size"]) * 1e6,
            "task_kind": np.random.choice(task_config["task_kinds"]),
            "is_safe": np.random.choice([0, 1],
                                        p=[task_config["safe_measurement"][0], task_config["safe_measurement"][1]]),
        }

    @staticmethod
    def generate_random_dag(num_nodes, ):
        dag = nx.DiGraph()
        nodes = [i + 1 for i in range(num_nodes)]
        dag.add_nodes_from(nodes)

        available_parents = {node: list(nodes[:i]) for i, node in enumerate(nodes)}
        for i in range(2, num_nodes + 1):
            num_parents = min(random.randint(1, min(i, jobs_config['max_num_parents_dag'])), len(available_parents[i]))
            parent_nodes = random.sample(available_parents[i], num_parents)
            dag.add_edges_from((parent_node, i) for parent_node in parent_nodes)
            available_parents[i] = list(nodes[:i])

        return dag

    @staticmethod
    def _load_csv(path, fallback_method):
        if not os.path.exists(path):
            fallback_method()
        df = pd.read_csv(path)
        if 'tasks_ID' in df.columns:
            df["tasks_ID"] = df["tasks_ID"].apply(lambda x: ast.literal_eval(x))
        if 'predecessors' in df.columns:
            df["predecessors"] = df["predecessors"].apply(lambda x: ast.literal_eval(x))
        if 'successors' in df.columns:
            df["successors"] = df["successors"].apply(lambda x: ast.literal_eval(x))
        if 'voltages_frequencies' in df.columns:
            df["voltages_frequencies"] = df["voltages_frequencies"].apply(lambda x: ast.literal_eval(x))
        if 'acceptable_tasks' in df.columns:
            df["acceptable_tasks"] = df["acceptable_tasks"].apply(lambda x: ast.literal_eval(x))
        return df.to_dict(orient='records')

    @classmethod
    def generate_random_device(cls):
        device_type = np.random.choice(["iot", "mec"])
        device_config = devices_config[device_type]
        cpu_cores = int(np.random.choice(device_config["num_cores"]))
        device = {
            "type": device_type,
            "num_cores": cpu_cores,
            "voltages_frequencies": [[device_config["voltage_frequencies"][i]
                                      for i in np.random.choice(len(device_config['voltage_frequencies']), size=3,
                                                                replace=False)]
                                     for _ in range(cpu_cores)
                                     ],
            "ISL": cls._generate_value(device_config["isl"]),
            "capacitance": np.random.uniform(*device_config["capacitance"]) * 1e-9,
            "powerIdle": float(np.random.choice(device_config["powerIdle"])) * 1e-6,
            "battery_capacity": cls._generate_value(device_config["battery_capacity"]) * 1e6,
            "acceptable_tasks": list(np.random.choice(jobs_config["task"]["task_kinds"],
                                                      size=np.random.randint(3, 4),
                                                      replace=False)),
            "is_safe": int(np.random.choice([0, 1], p=[device_config["safe"][0], device_config["safe"][1]])),
            "maxQueue":device_config['maxQueue'],
        }
        return device

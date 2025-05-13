import random
from configs import environment_config
from collections import deque


class WindowManager:
    def __init__(self, state, manager, config=environment_config):
        self.state = state
        self.__pool = manager.list()  # Keep as manager.list but use differently
        self.__pool_deque = deque()  # Local cache for better performance

        self.__max_jobs = config['window']["max_jobs"]
        self.__window_size = config['window']["size"]
        self.current_cycle = config['window']["clock"]
        self.__cycle = config['window']["clock"]
        self.__head_index = 0

    def run(self):
        # OPTIMIZATION: Early return and simplified logic
        if len(self.state.jobs) > 5:
            self.state.task_window = []
            return

        if self.current_cycle != self.__cycle:
            self.current_cycle += 1
            self.state.task_window = []
        else:
            self.current_cycle = 0
            self.state.task_window = self.get_window()

    def get_window(self):
        # OPTIMIZATION: Use local deque for better performance
        if not self.__pool_deque:
            if not self.__pool:
                self.__pool = self.__slice()
            self.__pool_deque.extend(self.__pool)
            self.__pool = []  # Clear shared pool

        window_size = min(len(self.__pool_deque), self.__window_size)
        return [self.__pool_deque.popleft() for _ in range(window_size)]

    def __slice(self):
        # OPTIMIZATION: Batch process tasks
        slice_end = self.__head_index + self.__max_jobs
        sliced_jobs = self.state.db_jobs[self.__head_index:slice_end]
        self.__head_index = slice_end

        # Flatten task IDs in one go
        selected_tasks = [
            task_id
            for job in sliced_jobs
            for task_id in job["tasks_ID"]
        ]
        random.shuffle(selected_tasks)
        return selected_tasks

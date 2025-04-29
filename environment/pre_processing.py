from collections import defaultdict
from configs import environment_config

class Preprocessing:
    def __init__(self, state, manager, config=environment_config):
        # OPTIMIZATION [PP-1]: Streamlined initialization and data structures
        self.state = state
        self.active_jobs = manager.dict()
        self.assigned_jobs = manager.list()
        self.queue = manager.list()
        self.max_jobs = config['multi_agent']
        
        # OPTIMIZATION [PP-2]: Cache for frequent lookups
        self._mobility_cache = manager.dict()  # Shared cache for task mobility
        self._task_job_cache = manager.dict()  # Cache task->job mapping

    def run(self):
        # OPTIMIZATION [PP-3]: Combined update operations
        self.update_active_jobs()
        self.process()

    def assign_job(self):
        # OPTIMIZATION [PP-4]: Efficient job assignment with proper synchronization
        for job in self.active_jobs.keys():
            if job not in self.assigned_jobs:
                self.assigned_jobs.append(job)
                return job
        return None

    def update_active_jobs(self):
        # OPTIMIZATION [PP-5]: Batch process job updates
        state_jobs = self.state.jobs
        
        # Add new jobs and update existing ones
        for job_ID, job_data in state_jobs.items():
            self.active_jobs[job_ID] = job_data
            
        # Remove completed jobs
        completed_jobs = []
        for job_ID, job in self.active_jobs.items():
            if job_ID not in state_jobs or \
               len(job['finishedTasks']) + len(job["runningTasks"]) == job["task_count"]:
                completed_jobs.append(job_ID)
        
        # Batch remove completed jobs
        for job_ID in completed_jobs:
            if job_ID in self.active_jobs:
                del self.active_jobs[job_ID]

    def process(self):
        # OPTIMIZATION [PP-6]: Efficient task window processing
        current_queue = set(self.queue)  # Convert to set for O(1) lookups
        db_tasks = self.state.db_tasks
        
        # Process new tasks from window
        for task_id in self.state.task_window:
            if task_id not in current_queue:  # Only process if not already in queue
                task = db_tasks[task_id]
                if task['pred_count'] <= 0:
                    current_queue.add(task_id)
                    # Cache task->job mapping
                    self._task_job_cache[task_id] = task['job_id']
        
        # Update queue with sorted tasks
        if current_queue:
            sorted_tasks = self._sort_by_mobility(list(current_queue))
            self.queue[:] = sorted_tasks

    def _sort_by_mobility(self, task_ids):
        # OPTIMIZATION [PP-7]: Efficient mobility sorting with caching
        mobility_dict = {}
        db_tasks = self.state.db_tasks
        
        for task_id in task_ids:
            if task_id not in self._mobility_cache:
                self._mobility_cache[task_id] = len(db_tasks[task_id]['successors'])
            mobility_dict[task_id] = self._mobility_cache[task_id]
        
        return sorted(task_ids, key=lambda x: mobility_dict[x])

    def get_agent_queue(self):
        # OPTIMIZATION [PP-8]: Efficient queue distribution
        job_queues = defaultdict(list)
        active_job_ids = set(self.active_jobs.keys())
        
        # Build queues in single pass
        for task_id in self.queue:
            job_id = self._task_job_cache.get(task_id)
            if not job_id:
                job_id = self.state.db_tasks[task_id]['job_id']
                self._task_job_cache[task_id] = job_id
                
            if job_id in active_job_ids:
                job_queues[job_id].append(task_id)
        
        # Convert defaultdict to regular dict for return
        return dict(job_queues)
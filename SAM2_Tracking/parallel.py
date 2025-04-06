import os
import multiprocessing
import functools
import parallel_utils as pu

num_workers = 3  # Number of worker processes

# Get the set of core IDs for each core
affinity_set = os.sched_getaffinity(0)

# Sort core IDs
available_cores = sorted(list(affinity_set))

# Ge the number of available cores
num_cores = len(available_cores)  

pu.checks(num_cores, num_workers)

core_assignments = pu.get_core_assignments(num_workers, available_cores)

data = [1, 2, 3, 4, 5, 6, 7, 8, 9 , 10]

# Create the partial function (allows us to pass in additional inputs)
partial_func = functools.partial(pu.worker_function, num_workers=num_workers)

lock = multiprocessing.Lock()
counter = multiprocessing.Value('i', 0) # Rank counter starts at 0

with multiprocessing.Pool(processes=num_workers, initializer=pu.init_worker, initargs=(core_assignments, lock, counter)) as pool:
    pool.map(partial_func, data)



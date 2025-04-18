import os
import psutil
import torch 
import time
import gc

def checks(num_cores, num_workers):
    # Check to see if the number of cores can be evenly distributed
    if num_cores % num_workers != 0:
        print(
            f"Warning: {num_cores} cores do not divide evenly among {num_workers} workers. "
            f"Load might be slightly unbalanced. "
            f"({num_cores % num_workers} worker(s) will have one extra core). "
            f"Consider adjusting the number of workers or allocated cores for perfect balance."
        )

    # Check to see if the number of workers exceeds the number of cores 
    if num_workers > num_cores:
        raise RuntimeError(
            f"Error: Number of workers ({num_workers}) exceeds the number of available cores ({num_cores}). "
            f"Please reduce the number of workers or increase the number of cores.")

def get_core_assignments(num_workers):

    # Get the set of core IDs for each core
    affinity_set = os.sched_getaffinity(0)

    # Sort core IDs
    available_cores = sorted(list(affinity_set))

    # Get the number of available cores
    num_cores = len(available_cores)  

    checks(num_cores, num_workers)

    # Create core assignments in a round-robin fashion 
    core_assignments = [[] for _ in range(num_workers)]
    for i, core_id in enumerate(available_cores):
        # Assign core 'core_id' to worker 'i % num_workers'
        worker_index = i % num_workers
        core_assignments[worker_index].append(core_id)

    return core_assignments

def set_process_affinity(cores):
    process = psutil.Process(os.getpid())
    process.cpu_affinity(cores)
    print(f"Process {process.pid} is running on cores {cores}")

def init_worker(all_assignments, lock, counter):
    
    # Make worker_rank modifiable within this function's scope (for this process)
    global worker_rank
    
    # Also store the args globally if needed elsewhere, but primarily used here
    global initializer_lock, initializer_counter, initializer_core_assignments
    initializer_lock = lock
    initializer_counter = counter
    initializer_core_assignments = all_assignments

    # Assign the rank to the process 
    with lock:
        worker_rank = counter.value
        counter.value += 1

    # Assign cores based on the obtained rank
    my_cores = all_assignments[worker_rank]
    
    # set_process_affinity will now print the rank as well
    set_process_affinity(my_cores)


def worker_function(item, num_workers, device_input):

    print(f"process: {os.getpid()} processing item {item}")
    # print(f"1.0/num_workers: {1.0/num_workers}")
    device = torch.device(device_input)

    # free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(device)
    # total_memory_mib = total_memory_bytes / (1024 ** 2)
    # # free_memory_mib = free_memory_bytes / (1024 ** 2)
    # print(f"total_memory_mib: {total_memory_mib}")
    # # print(f"free_memory_mib: {free_memory_mib}")

    torch.cuda.set_per_process_memory_fraction(1.0/num_workers, device=device)
    a = torch.ones((13, 10000, 10000))
    a = a.to(device)
    a *= item

    # del a 
    # torch.cuda.empty_cache()
    # gc.collect()
    time.sleep(10)  # Simulate work

import torch
import pynvml


def select_device():
    def get_mem(device):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        mem = 0
        num_process = 0
        for p in procs:
            mem += p.usedGpuMemory / (1024 * 1024)
            num_process += 1
        return device, num_process, mem

    usage = {}
    gpu_num = torch.cuda.device_count()
    print('\nUsage of GPU:\n----------------------------------------------------')
    for idx in range(gpu_num):
        print(torch.cuda.list_gpu_processes('cuda:%d' % idx))
        if idx == gpu_num - 1:
            print('----------------------------------------------------')
        device, num_process, mem = get_mem(idx)
        usage[device] = (num_process, mem)
    gpu_id = sorted(usage.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)[-1][0]
    print('Selected device: cuda:%s\n\n' % gpu_id)
    return 'cuda:%d' % gpu_id




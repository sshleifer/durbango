import pyarrow.plasma as plasma
import psutil
import subprocess
import numpy as np
import gc
import os
import time
from fire import Fire
import tempfile

def calc_mem():
    return int(psutil.virtual_memory().used / (1024**2))
GB200 = (1024 ** 3) * 200
def start_plasma_store(
    path='/tmp/plasma', nbytes: int = GB200
) -> subprocess.Popen:
    # best practice is to allocate more space than we need. The limitation seems to be the size of /dev/shm
    _server = subprocess.Popen(["plasma_store", "-m", str(nbytes), "-s", path])



def benchmark_plasma():
    arr = np.zeros((int(1e8), 3))
    time.sleep(2)
    start = calc_mem()
    tstart = time.time()
    def print_msg(msg):
        print(f'{msg}: {calc_mem() - start } MB. t={time.time()-tstart:.2f}')

    client = plasma.connect('/tmp/plasma')
    print_msg('have client')
    oid = client.put(arr)
    client.disconnect()
    print_msg('done put')
    client = plasma.connect('/tmp/plasma')
    gc.collect()
    print_msg('deleted original array')
    import ipdb; ipdb.set_trace()
    #arrs = [client.get(oid) for _ in range(10)]
    a2 = client.get(oid)
    print_msg('read full array')
    #a2[4] = a2[3]
    print(type(a2))
    del a2
    gc.collect()
    print_msg('del array')
    a3 = client.get(oid)[0]
    print_msg('read one entry')
    gc.collect()
    print_msg('collect, done')

def array_to_memmap(array, filename):
    fp = np.memmap(filename,mode='write', dtype=array.dtype, shape=array.shape)
    fp[:] = array[:]  # copy
    fp.flush()
    return fp

def benchmark_mmap():
    arr = np.zeros((int(1e8), 3))
    time.sleep(2)
    start = calc_mem()
    tstart = time.time()
    def print_msg(msg):
        print(f'{msg}: {calc_mem() - start } MB. t={time.time()-tstart:.2f}')

    f = tempfile.NamedTemporaryFile()
    fp = np.memmap(f.name, mode='write', dtype=arr.dtype, shape=arr.shape)
    print_msg('Init memmap')
    fp[:] = arr[:]  # copy
    gc.collect()
    print_msg('Copy to memmap')
    fp.flush()
    print_msg('flush memmap')
    gc.collect()
    print_msg('collect')
    one_entry = fp[2]
    print_msg('read one')
    #f.close()
    #del fp, one_entry
    gc.collect()
    print_msg('tried to delete')


def run_all():

    print('MMAP')
    benchmark_mmap()
    print('\n\nPlasma')
    benchmark_plasma()


if __name__ == '__main__':
    Fire(run_all)

import argparse
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import subprocess
from pyarrow import plasma
NO_LOCK = os.getenv("NO_LOCK", False)
GB200 = (1024 ** 3) * 200
DEFAULT_PLASMA_PATH = '/tmp/plasma'
import hashlib
from filelock import FileLock
import time

class PlasmaView:
    def __init__(
        self, array, split_path: str, object_num: int, path=DEFAULT_PLASMA_PATH
    ):
        assert split_path is not None
        self.path = path
        self.split_path = split_path
        self.object_id = self.get_object_id(self.split_path, object_num)
        self._client = None  # Initialize lazily for pickle, (TODO(SS): needed?)
        self._n = None
        self.use_lock = not NO_LOCK
        if array is not None:
            try:
                self.client.put(array, object_id=self.object_id)
                self.log("PUT")
            except plasma.PlasmaObjectExists:
                self.log("PlasmaObjectExists")

    @property
    def client(self):
        if self._client is None:
            self._client = plasma.connect(self.path, num_retries=200)
        return self._client
        # return self._client

    @property
    def array(self):
        """Fetch a read only view of a np array, stored in plasma."""
        self.log("GET")
        if self.use_lock:
            with FileLock("/tmp/plasma_read_lock"):
                ret = self.client.get(self.object_id)

        else:
            ret = self.client.get(self.object_id)
        self.log("GOT")
        return ret

    def log(self, msg: str) -> None:
        print(f"pid: {os.getpid()}, id: {self.object_id}, lock: {self.use_lock}: {msg}")
        #pass

    @staticmethod
    def int_to_bytes(x: int) -> bytes:
        return x.to_bytes(
            (x.bit_length() + 7) // 8, "big"
        )  # https://tinyurl.com/56j5964v

    @staticmethod
    def get_object_id(split_path: str, object_num: int) -> plasma.ObjectID:
        hash = hashlib.blake2b(bytes(split_path, "utf-8"), digest_size=20)
        hash.update(object_num.to_bytes(4, byteorder="big"))
        return plasma.ObjectID(hash.digest())

    @staticmethod
    def get_object_id_arr_unused(arr) -> plasma.ObjectID:
        """Just hash the shape"""
        # TODO(SS): delete if useless
        hash = hashlib.blake2b(b"0", digest_size=20)
        for dim in arr.shape:
            hash.update(dim.to_bytes(4, byteorder="big"))
        return plasma.ObjectID(hash.digest())

    def __getstate__(self):
        """Called on pickle save, I believe"""
        self.client.disconnect()
        self.log('get state')
        if getattr(self, '_client', None) is not None:
            self._client.disconnect()
            self._client = None

        state = self.__dict__.copy()
        state["_client"] = None
        assert 'object_id' in state
        return state

    # def __setstate__(self, state):
    #     """Called on pickle load, I believe"""
    #
    #     self.__dict__.update(state)
    #     self.log('set state')
    #     # self.client = plasma.connect(self.path, num_retries=200)

    def __del__(self):
        if self._client is not None: self._client.disconnect()
        self._client = None

    def __len__(self):
        """Save reads by caching len"""
        if self._n is None:
            self._n = len(self.array)
        return self._n


def start_plasma_store(path=DEFAULT_PLASMA_PATH, nbytes: int = GB200) -> subprocess.Popen:
    # best practice is to allocate more space than we need. The limitation seems to be the size of /dev/shm
    _server = subprocess.Popen(["plasma_store", "-m", str(nbytes), "-s", path])
    plasma.connect(path, num_retries=200)  # If we can't connect we fail immediately
    return _server


import torch

class PlasmaDataset(Dataset):

    def __init__(self, array, split_path):
        super().__init__()
        self.pv = PlasmaView(array, split_path, 0)

    def __getitem__(self, item):
        return torch.tensor(self.pv.array[item])

    def __len__(self):
        return len(self.pv)


def train(num_workers):
    dtrain = np.random.rand(1000, 10)
    dval = np.random.rand(100, 10)
    t0 = time.time()
    train_ds = PlasmaDataset(dtrain, 'train')
    val_ds = PlasmaDataset(dval, 'val')
    train_dl = DataLoader(train_ds, batch_size=4, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=4, num_workers=num_workers)
    for batch in train_dl:
        lens = batch[0][0]
    for batch in val_dl:
        lens = batch[0][0]
    print(f'DONE: {time.time() - t0:.2f} seconds')


def main():
    server = start_plasma_store()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, type=int)
    args = parser.parse_args()
    #parser.add_argument("--dimension", type=int, default=1024, help="Size of each key")
    train(args.num_workers)
    server.kill()


if __name__ == '__main__':
    main()

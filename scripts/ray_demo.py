import argparse
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import time
import ray
import torch

class RayDataset(Dataset):

    def __init__(self, array, split_path):
        super().__init__()
        self.ref = ray.put(array)

    def __getitem__(self, item):
        arr = ray.get(self.ref)[item]
        return torch.tensor(arr)

    def __len__(self):
        return len(ray.get(self.ref))


def train(num_workers):
    dtrain = np.random.rand(1000, 10)
    dval = np.random.rand(100, 10)
    t0 = time.time()
    train_ds = RayDataset(dtrain, 'train')
    val_ds = RayDataset(dval, 'val')
    train_dl = DataLoader(train_ds, batch_size=4, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=4, num_workers=num_workers)
    for batch in train_dl:
        lens = batch[0][0]
    for batch in val_dl:
        lens = batch[0][0]
    print(f'DONE: {time.time() - t0:.2f} seconds')


def main():

    #server = start_plasma_store()
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=0, type=int)
    args = parser.parse_args()
    #parser.add_argument("--dimension", type=int, default=1024, help="Size of each key")
    train(args.num_workers)
    ray.shutdown()
    #server.kill()


if __name__ == '__main__':
    main()

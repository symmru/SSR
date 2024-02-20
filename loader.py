import numpy as np
from glob import glob
import os
import random
from torch.utils.data import Dataset, DataLoader

# sphere mesh size at different levels
nv_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842, 655362, 2621442]

nv_partial_sphere = [12, 42, 45, 54, 84, 192, 600, 2184, 8424, 33192, 131880]

class MyDataLoader(Dataset):
    """Data loader for 2D3DS dataset."""

    def __init__(self, data_dir, sp_level, min_level, in_ch=3):
        self.in_ch = in_ch
        self.data_dir=data_dir
        #self.nv = nv_sphere[sp_level]
        self.nv = nv_partial_sphere[sp_level]
        self.train_nv = nv_partial_sphere[min_level]

        self.flist = []
        
        file_format = os.path.join(data_dir, "*.npz")
        self.flist += sorted(glob(file_format))


    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        fname = self.data_dir+'/%d.npz'%idx
        data = np.load(fname)['label'].T[:self.in_ch,:self.train_nv]
        label = np.load(fname)['label'].T[:self.in_ch,:self.nv]
        return data, label

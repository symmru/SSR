from torch import nn
from ops import MeshConv, ResBlock1, ResBlock2, MeshShuffle_SSR1, MeshShuffle_SSR2, MeshConv_transpose

import torch.nn.functional as F
import os
import torch
import pickle


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
        super().__init__()
        # mesh_file = mesh_folder + "/mesh_"+ str(level) + "_0.npz"
        mesh_file = os.path.join(mesh_folder, "partial_icosphere_{}.pkl".format(level))
        self.up = MeshConv_transpose(in_ch, in_ch, mesh_file, stride=2)
        self.conv = ResBlock1(in_ch, out_ch, out_ch, level, False, mesh_folder,True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class Transpose_Model(nn.Module):
    def __init__(self, mesh_folder="mesh_files", in_ch = 3, out_ch = 3, max_level=9, min_level=7, fdim=2):
        super().__init__()
        mf_in = "mesh_files/partial_icosphere_{}.pkl".format(min_level)
        mf_out = "mesh_files/partial_icosphere_{}.pkl".format(max_level)
        self.in_conv = MeshConv(in_ch, fdim, mf_in, stride=1, bias=True)
        self.in_bn = nn.BatchNorm1d(fdim)
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(self.in_conv,self.in_bn,self.relu)
        self.block1 = Up(fdim, 16*fdim, level=min_level+1, mesh_folder=mesh_folder, bias=True)
        self.block2 = Up(16*fdim, 16*fdim, level=min_level+2, mesh_folder=mesh_folder, bias=True)
        self.out_conv = MeshConv(16*fdim, out_ch, mesh_file = mf_out, stride = 1, bias=True)
        self.out_bn = nn.BatchNorm1d(out_ch)
        self.out_block = nn.Sequential(self.out_conv, self.out_bn, self.relu)

    def forward(self, x):
        x = self.in_block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.out_block(x)
        return x

class SuperResolution(nn.Module):
    def __init__(self, mesh_folder, in_ch, out_ch, max_level=9, min_level=7, fdim=16, method="SSR2"):
        super().__init__()
        self.mesh_folder = mesh_folder
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.diff = max_level-min_level

        self.in_conva = MeshConv(in_ch, fdim, self.__meshfile(min_level), stride=1, bias=True)
        self.in_bn = nn.BatchNorm1d(fdim)
        self.relu = nn.ReLU(inplace=True)
        self.in_block = nn.Sequential(self.in_conva, self.in_bn, self.relu)

        self.resblock1 = ResBlock2(fdim, 16*fdim, 16*fdim, min_level, False, mesh_folder, partial=True)
        if self.diff == 2:
            self.resblock2 = ResBlock2(16*fdim, 16*fdim, 16*fdim, min_level, False, mesh_folder, partial=True)
        elif self.diff == 3:
            self.resblock2 = ResBlock2(16*fdim, 64*fdim, 64*fdim, min_level, False, mesh_folder, partial=True)

        if method == "SSR2":
            self.shuffle1 = MeshShuffle_SSR2(self.__meshfile(min_level+1))
            self.shuffle2 = MeshShuffle_SSR2(self.__meshfile(min_level+2))
            if self.diff == 3:
                self.shuffle3 = MeshShuffle_SSR2(self.__meshfile(min_level+3))
        elif method == "SSR1":
            self.shuffle1 = MeshShuffle_SSR1(self.__meshfile(min_level+1))
            self.shuffle2 = MeshShuffle_SSR1(self.__meshfile(min_level+2))
            if self.diff == 3:
                self.shuffle3 = MeshShuffle_SSR1(self.__meshfile(min_level+3))

        self.out_conva = MeshConv(fdim, out_ch, self.__meshfile(max_level), stride=1, bias=True)
        self.out_bn = nn.BatchNorm1d(out_ch)
        self.out_block = nn.Sequential(self.out_conva, self.out_bn)

    def forward(self, x):
        x = self.in_block(x)
        x_ = x
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.diff == 2:
            x = torch.cat([x_]*16, dim=1) + x
        elif self.diff == 3:
            x = torch.cat([x_]*64, dim=1) + x
        x = self.shuffle1(x)
        x = self.shuffle2(x)
        if self.diff == 3:
            x = self.shuffle3(x)
        x = self.out_block(x)
        return x

    def __meshfile(self, i):
        return os.path.join(self.mesh_folder, "partial_icosphere_{}.pkl".format(i))

class TestMeshShuffle(nn.Module):
    def __init__(self, mesh_folder, in_ch, out_ch, max_level=5, min_level=0, fdim=16):
        super().__init__()
        self.mesh_folder = mesh_folder
        self.min_level = min_level
        self.in_conv = MeshConv(in_ch, fdim, self.__meshfile(min_level), stride=1, bias=True)        
        del fdim
        del max_level
        del in_ch
        del out_ch
        self.shuffle = MeshShuffle(self.__meshfile(min_level+1))
        
    def forward(self, x):
        x = torch.cat([x] * 4, dim=1)
        x = self.shuffle(x)
        return x

    def __meshfile(self, i):
        return os.path.join(self.mesh_folder, "partial_icosphere_{}.pkl".format(i))

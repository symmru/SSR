import math
import pickle, gzip
import os

import torch
from torch import nn
from torch.nn.parameter import Parameter
from scipy import sparse

import numpy as np

def sparse2tensor(m):
    """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
    if isinstance(m, sparse.csc.csc_matrix):
        m = m.tocoo()
    assert(isinstance(m, sparse.coo.coo_matrix))
    i = torch.LongTensor([m.row, m.col])
    v = torch.FloatTensor(m.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(m.shape))


def spmatmul(den, sp):
    """
    den: Dense tensor of shape batch_size x in_chan x #V
    sp : Sparse tensor of shape newlen x #V
    """
    batch_size, in_chan, nv = list(den.size())
    new_len = sp.size()[0]
    den = den.permute(2, 1, 0).contiguous().view(nv, -1)
    res = torch.spmm(sp, den).view(new_len, in_chan, batch_size).contiguous().permute(2, 1, 0)
    return res


def spmatmul_new(den, sp):
    batch_size, in_chan, nv = list(den.size())
    new_len = sp.size()[0]
    den = den.permute(2, 1, 0).contiguous().view(nv, -1)
    res = torch.spmm(sp, den).view(new_len, in_chan, batch_size).permute(2, 0, 1).contiguous()
    return res


def EW_NS_to_sparse(m):
    """
    m is either EW or NS, #F x 3 dimension
    outputs a diagonal sparse matrix in #F x 3#F
    """
    num_faces = m.shape[0]
    data = m.flatten()
    row_range = np.arange(num_faces)
    col_range = np.arange(3)
    col_idxs, row_idxs = np.meshgrid(col_range, row_range)
    
    # 3 groups of F
    col_idxs = (col_idxs * num_faces + np.arange(num_faces)[:, None])    

    col_idxs = np.reshape(col_idxs, (-1))
    row_idxs = np.reshape(row_idxs, (-1))
    sp_m = sparse.coo_matrix((data, (row_idxs, col_idxs)), shape=(num_faces, 3*num_faces))

    return sp_m

class _MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, mesh_file, stride=1, bias=True):
        assert stride in [1, 2]
        super(_MeshConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.ncoeff = 4
        #self.coeffs = Parameter(torch.Tensor(out_channels, in_channels, self.ncoeff))
        self.coeffs = Parameter(torch.Tensor(self.ncoeff, in_channels, out_channels))
        self.set_coeffs()
        # load mesh file
        pkl = pickle.load(open(mesh_file, "rb"))
        self.pkl = pkl
        self.nv = self.pkl['V'].shape[0]

        G = sparse2tensor(pkl['G'])  # gradient matrix V->F, 3#F x #V
        NS = torch.tensor(pkl['NS'], dtype=torch.float32)  # north-south vector field, #F x 3
        EW = torch.tensor(pkl['EW'], dtype=torch.float32)  # east-west vector field, #F x 3
        self.register_buffer("G", G)
        self.register_buffer("NS", NS)
        self.register_buffer("EW", EW)
        
    def set_coeffs(self):
        n = self.in_channels * self.ncoeff
        stdv = 1. / math.sqrt(n)
        self.coeffs.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

class MeshConv(_MeshConv):
    def __init__(self, in_channels, out_channels, mesh_file, stride=1, bias=True):
        super(MeshConv, self).__init__(in_channels, out_channels, mesh_file, stride, bias)
        pkl = self.pkl
        if stride == 2:
            self.nv_prev = pkl['nv_prev']
            L_coo = pkl['L'].tocsr()[:self.nv_prev].tocoo()
            L = sparse2tensor(L_coo) # laplacian matrix V->V
        else: # stride == 1
            self.nv_prev = pkl['V'].shape[0]
            L_coo = pkl['L'].tocoo()
            L = sparse2tensor(L_coo)
        self.register_buffer("L", L)

        NS_sparse = EW_NS_to_sparse(pkl['NS'])
        EW_sparse = EW_NS_to_sparse(pkl['EW'])
 
        NS_grad_op = pkl['F2V'] * NS_sparse * pkl['G']
        NS_grad_op_coo = NS_grad_op.tocoo() 
        NS_grad_op = sparse2tensor(NS_grad_op_coo) 
        self.register_buffer("NS_grad_op", NS_grad_op)


        EW_grad_op = pkl['F2V'] * EW_sparse * pkl['G'] 
        EW_grad_op_coo = EW_grad_op.tocoo() 
        EW_grad_op = sparse2tensor(EW_grad_op_coo) 
        self.register_buffer("EW_grad_op", EW_grad_op)

    def forward(self, input):
        # less memory, fast, but not sure if it is equivalent
        laplacian = spmatmul_new(input, self.L)
        identity = input[..., :self.nv_prev].permute(0, 2, 1)

        grad_vert_ew = spmatmul_new(input, self.EW_grad_op)
        grad_vert_ns = spmatmul_new(input, self.NS_grad_op)

        # identity
        out = torch.matmul(identity, self.coeffs[0,:,:])
        # laplacian
        out += torch.matmul(laplacian, self.coeffs[1,:,:])
        # grad_vert_ew 
        out += torch.matmul(grad_vert_ew, self.coeffs[2,:,:])
        # grad_vert_ns 
        out += torch.matmul(grad_vert_ns, self.coeffs[3,:,:])
        out += self.bias
        out = out.permute(0, 2, 1)
        return out



class MeshConv_transpose(_MeshConv):
    def __init__(self, in_channels, out_channels, mesh_file, stride=2, bias=True):
        assert(stride == 2)
        super(MeshConv_transpose, self).__init__(in_channels, out_channels, mesh_file, stride, bias)
        pkl = self.pkl
        self.nv_prev = self.pkl['nv_prev']
        self.nv_pad = self.nv - self.nv_prev
        L = sparse2tensor(pkl['L'].tocoo()) # laplacian matrix V->V
        F2V = sparse2tensor(pkl['F2V'].tocoo()) # F->V, #V x #F
        self.register_buffer("L", L)
        self.register_buffer("F2V", F2V)
        
        NS_sparse = EW_NS_to_sparse(pkl['NS'])
        EW_sparse = EW_NS_to_sparse(pkl['EW'])

        NS_grad_op = pkl['F2V'] * NS_sparse * pkl['G']
        NS_grad_op = sparse2tensor(NS_grad_op.tocoo())
        self.register_buffer("NS_grad_op", NS_grad_op)

        EW_grad_op = pkl['F2V'] * EW_sparse * pkl['G']
        EW_grad_op = sparse2tensor(EW_grad_op.tocoo())
        self.register_buffer("EW_grad_op", EW_grad_op)

    def forward(self, input):
        # pad input with zeros up to next mesh resolution
        ones_pad = torch.ones(*input.size()[:2], self.nv_pad).to(input.device)
        input = torch.cat((input, ones_pad), dim=-1)

        # less memory, fast, and confirmed it is equivalent
        laplacian = spmatmul_new(input, self.L)
        identity = input.permute(0, 2, 1)

        grad_vert_ew = spmatmul_new(input, self.EW_grad_op)
        grad_vert_ns = spmatmul_new(input, self.NS_grad_op)

        # identity
        out = torch.matmul(identity, self.coeffs[0,:,:])
        # laplacian
        out += torch.matmul(laplacian, self.coeffs[1,:,:])
        # grad_vert_ew 
        out += torch.matmul(grad_vert_ew, self.coeffs[2,:,:])
        # grad_vert_ns 
        out += torch.matmul(grad_vert_ns, self.coeffs[3,:,:])
        out += self.bias
        out = out.permute(0, 2, 1)
        return out
        
class ResBlock1(nn.Module):
    def __init__(self, in_chan, neck_chan, out_chan, level, coarsen, mesh_folder, partial=False):
        super().__init__()
        l = level-1 if coarsen else level
        self.coarsen = coarsen
        
        if partial:
            mesh_file = os.path.join(mesh_folder, "partial_icosphere_{}.pkl".format(l))
        else:
            mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(l))

        self.conv1 = nn.Conv1d(in_chan, neck_chan, kernel_size=1, stride=1)
        self.conv2 = MeshConv(neck_chan, neck_chan, mesh_file=mesh_file, stride=1)
        self.conv3 = nn.Conv1d(neck_chan, out_chan, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(neck_chan)
        self.bn2 = nn.BatchNorm1d(neck_chan)
        self.bn3 = nn.BatchNorm1d(out_chan)
        self.nv_prev = self.conv2.nv_prev
        #self.down = DownSamp(self.nv_prev)
        self.diff_chan = (in_chan != out_chan)

        if coarsen:
            # self.seq1 = nn.Sequential(self.down, self.conv1, self.bn1, self.relu, 
            #                           self.conv2, self.bn2, self.relu, 
            #                           self.conv3, self.bn3)
            self.seq1 = nn.Sequential(self.conv1, self.down, self.bn1, self.relu, 
                                      self.conv2, self.bn2, self.relu, 
                                      self.conv3, self.bn3)
        else:
            self.seq1 = nn.Sequential(self.conv1, self.bn1, self.relu, 
                                      self.conv2, self.bn2, self.relu, 
                                      self.conv3, self.bn3)

        if self.diff_chan or coarsen:
            self.conv_ = nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=1)
            self.bn_ = nn.BatchNorm1d(out_chan)
            if coarsen:
                self.seq2 = nn.Sequential(self.conv_, self.down, self.bn_)
            else:
                self.seq2 = nn.Sequential(self.conv_, self.bn_)

    def forward(self, x):
        if self.diff_chan or self.coarsen:
            x2 = self.seq2(x)
        else:
            x2 = x
        x1 = self.seq1(x)
        out = x1 + x2
        out = self.relu(out)
        return out

class ResBlock2(nn.Module):
    def __init__(self, in_chan, neck_chan, out_chan, level, coarsen, mesh_folder, partial=False):
        super().__init__()
        l = level-1 if coarsen else level
        self.coarsen = coarsen
        
        if partial:
            mesh_file = os.path.join(mesh_folder, "partial_icosphere_{}.pkl".format(l))
        else:
            mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(l))

        self.conv1 = nn.Conv1d(in_chan, neck_chan, kernel_size=1, stride=1)
        self.conv2a = MeshConv(neck_chan, neck_chan, mesh_file=mesh_file, stride=1)
        self.conv2b = MeshConv(neck_chan, neck_chan, mesh_file=mesh_file, stride=1)
        self.conv3 = nn.Conv1d(neck_chan, out_chan, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(neck_chan)
        self.bn2 = nn.BatchNorm1d(neck_chan)
        self.bn3 = nn.BatchNorm1d(out_chan)
        self.nv_prev = self.conv2a.nv_prev
        #self.down = DownSamp(self.nv_prev)
        self.diff_chan = (in_chan != out_chan)

        if coarsen:
            # self.seq1 = nn.Sequential(self.down, self.conv1, self.bn1, self.relu, 
            #                           self.conv2, self.bn2, self.relu, 
            #                           self.conv3, self.bn3)
            self.seq1 = nn.Sequential(self.conv1, self.down, self.bn1, self.relu, 
                                      self.conv2, self.bn2, self.relu, 
                                      self.conv3, self.bn3)
        else:
            self.seq1 = nn.Sequential(self.conv1, self.bn1, self.relu, 
                                      self.conv2a, self.conv2b, self.bn2, self.relu, 
                                      self.conv3, self.bn3)

        if self.diff_chan or coarsen:
            self.conv_ = nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=1)
            self.bn_ = nn.BatchNorm1d(out_chan)
            if coarsen:
                self.seq2 = nn.Sequential(self.conv_, self.down, self.bn_)
            else:
                self.seq2 = nn.Sequential(self.conv_, self.bn_)

    def forward(self, x):
        if self.diff_chan or self.coarsen:
            x2 = self.seq2(x)
        else:
            x2 = x
        x1 = self.seq1(x)
        out = x1 + x2
        out = self.relu(out)
        return out


class DownSamp(nn.Module):
    def __init__(self, nv_prev):
        super().__init__()
        self.nv_prev = nv_prev

    def forward(self, x):
        return x[..., :self.nv_prev]

      
class MeshShuffleLIN(nn.Module):
    def __init__(self, mesh_file):
        super().__init__()
        pkl = pickle.load(open(mesh_file, "rb"))

        new_vert_map = torch.tensor(pkl['new_vert_map'], dtype=torch.long)
        self.register_buffer("new_vert_map", new_vert_map)

        self.coeffs = Parameter(torch.Tensor(4, 2, 8))
        self.set_coeffs()

    def set_coeffs(self):
        n = 4 * 2 * 8
        stdv = 1. / math.sqrt(n)
        self.coeffs.data.uniform_(-stdv, stdv)

    def forward(self, x):
      # [batch, fdim, ico_vert_vals_at_level]

      # Returns
      # [batch, fdim / 4, ico_vert_vals_at_level_plus_1]
      inputs = x[:, :, self.new_vert_map]
      n_ch = int(x.shape[1] / 4)
      inputs = inputs.view(x.shape[0], n_ch, -1, inputs.shape[2], 2)
      #outputs = torch.einsum("bcevd,ed->bcv", inputs, self.coeffs)

      soft_max = torch.nn.Softmax(dim=3)
      # w = soft_max(self.coeffs)
      w = self.coeffs
      attn = torch.einsum("bcevd,edf->bcvf", inputs, w)
      attn = soft_max(attn / 8)
      # print(attn.shape)
      attn = attn.view(*attn.shape[:3], 4, 2)
      outputs = torch.einsum("bcevd,bcved->bcv", inputs, attn)


      return outputs


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
        super().__init__()
        # mesh_file = mesh_folder + "/mesh_"+ str(level) + "_0.npz"
        mesh_file = os.path.join(mesh_folder, "partial_icosphere_{}.pkl".format(level))
        self.up = MeshConv_transpose(in_ch, in_ch, mesh_file, stride=2)
        self.conv = ResBlock(in_ch, out_ch, out_ch, level, False, mesh_folder,True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class MeshShuffle_SSR1(nn.Module):
    def __init__(self, mesh_file):
        super().__init__()
        pkl = pickle.load(open(mesh_file, "rb"))

        separated_src_idx = torch.tensor(pkl['separated_src_idx'], dtype=torch.long)
        unique = torch.tensor(pkl['unique'], dtype=torch.long)
        self.register_buffer("separated_src_idx", separated_src_idx)
        self.register_buffer("unique", unique)

        
    def forward(self, x):
      # [batch, fdim, ico_vert_vals_at_level]
      
      # Returns
      # [batch, fdim / 4, ico_vert_vals_at_level_plus_1]
      
      # Get sub-vertices
      n_ch = int(x.shape[1] / 4)
      inputs = [x[:, i*n_ch:(i+1)*n_ch, :] for i in range(4)]

      # Map the final three fdims to vertices in the icosahedron refined at the
      # next level.
      #
      # The mapping takes the average (of two source vertex values) at the input ico level
      # and maps them to a face at the next refinement level.
      face_vert_outputs = []
      for i in range(3):
        inp = inputs[i + 1] / 2.  # Average over two inputs.
        faces = self.separated_src_idx[i]
        v0 = faces[:, 0]
        v1= faces[:, 1]
        # Two vertices for each refined vertex, some of these are duplicates, which
        # will be filtered out later.
        face_vert_outputs.append(inp[:,:,v0] + inp[:,:,v1])
          
      # Deduplicate per-face value information to get per-vertex information for the
      # refined outputs.
      face_vert_outputs = torch.cat(face_vert_outputs, dim=2)

      # Gather args are input, dim, index
      f2v = self.unique
      face_vert_outputs = face_vert_outputs[:,:,f2v]
      #face_vert_outputs = spmatmul(face_vert_outputs, self.unique_mat)
      
      # The final output is the unmapped vertices (which remain the same across levels)
      # concatenated with the mapped vertices.
      full_output = torch.cat([inputs[0], face_vert_outputs], dim=2)
      
      return full_output


class MeshShuffle_SSR2(nn.Module):
    def __init__(self, mesh_file):
        super().__init__()
        pkl = pickle.load(open(mesh_file, "rb"))

        separated_src_idx = torch.tensor(pkl['separated_src_idx'], dtype=torch.long)
        unique = torch.tensor(pkl['unique'], dtype=torch.long)
        self.register_buffer("separated_src_idx", separated_src_idx)
        self.register_buffer("unique", unique)



    def forward(self, x):
        n_ch = int(x.shape[1] / 4)
        face_vert_outputs = []
        tmp = [x[:, i*n_ch:(i+1)*n_ch, :] for i in range(4)]

        part1 = (tmp[0]+tmp[1])/2
        part2 = (tmp[2]+tmp[3])/2



        for i in range(3):
            inp = part2/2
            faces = self.separated_src_idx[i]
            v0 = faces[:, 0]
            v1= faces[:, 1]
            # Two vertices for each refined vertex, some of these are duplicates, which
            # will be filtered out later.
            face_vert_outputs.append(inp[:,:,v0] + inp[:,:,v1])

        # Deduplicate per-face value information to get per-vertex information for the
        # refined outputs.
        face_vert_outputs = torch.cat(face_vert_outputs, dim=2)

        # Gather args are input, dim, index
        f2v = self.unique
        face_vert_outputs = face_vert_outputs[:,:,f2v]
        #face_vert_outputs = spmatmul(face_vert_outputs, self.unique_mat)
      
        # The final output is the unmapped vertices (which remain the same across levels)
        # concatenated with the mapped vertices.
        full_output = torch.cat([part1, face_vert_outputs], dim=2)          
      
        return full_output
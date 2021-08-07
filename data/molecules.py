import torch
import pickle
import torch.utils.data
import time
import os
import pandas as pd
import csv
import dgl
from scipy import sparse as sp
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import networkx as nx
import hashlib
from train.config import Config


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, cfg, chr, split):
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.split = split
        self.cell = cfg.cell
        self.chr = chr
        self.contact_data = None

        self.load_hic()
        self.create_hic_graphs()

        """
        data is a list of Molecule dict objects with following attributes
        
          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """

        self.graph_lists = []
        self.graph_labels = []
        self._prepare()

    def load_hic(self):
        data = pd.read_csv("%s%s/%s/hic_chr%s.txt" % (self.cfg.hic_path, self.cell, self.chr, self.chr), sep="\t",
                           names=['i', 'j', 'v'])

        data[['i', 'j']] = data[['i', 'j']] / self.cfg.resolution
        data[['i', 'j']] = data[['i', 'j']].astype('int64')

        chr_max = int(data["i"].max())

        self.contact_data = np.zeros((chr_max, chr_max))
        rows = np.array(data["i"]).astype(int)
        cols = np.array(data["j"]).astype(int)
        data = self.contact_prob(np.array(data["v"]))

        self.contact_data[rows, cols] = data
        self.contact_data[cols, rows] = data

    def create_hic_graphs(self):
        print("preparing graphs for the %s set..." % (self.split.upper()))
        window = self.cfg.num_nodes
        num_windows = int(np.ceil(self.contact_data.shape[0] / window))
        num_nodes_last_frame = int(self.contact_data.shape[0] % window)

        for i in range(num_windows):
            for j in range(num_windows):
                if i == num_windows - 1 and j == num_windows - 1:
                    window_data = self.contact_data[i * window:i * window + num_nodes_last_frame,
                                  j * window:j * window + num_nodes_last_frame]

                elif j == num_windows - 1:
                    window_data = self.contact_data[i * window:(i + 1) * window,
                                  j * window:j * window + num_nodes_last_frame]
                elif i == num_windows - 1:
                    window_data = self.contact_data[i * window:i * window + num_nodes_last_frame,
                                  j * window:j * window + num_nodes_last_frame]
                else:
                    window_data = self.contact_data[i * window:(i + 1) * window,
                                  j * window:(j + 1) * window]

                if window_data.all() == 0:
                    continue

                window_data = np.round(window_data, 2) * 100
                window_data = window_data.astype(int)
                window_data = ~np.all(window_data == 0, axis=1)
                window_data = ~np.all(window_data == 0, axis=0)
                num_nodes = window_data.shape[0]

                edge_list = (window_data != 0).nonzero()
                edge_idxs_in_adj = edge_list.split(1, dim=1)
                edge_features = window_data[edge_idxs_in_adj].reshape(-1).long()
                node_features = None

                # Create the DGL Graph
                g = dgl.DGLGraph()
                g.add_nodes(num_nodes)
                g.ndata['feat'] = node_features

                for src, dst in edge_list:
                    g.add_edges(src.item(), dst.item())
                g.edata['feat'] = edge_features

                self.graph_lists.append(g)
                self.graph_labels.append(torch.tensor(np.mean(window_data)))

        pass

    def contact_prob(self, values, delta=1e-10):
        coeff = np.nan_to_num(1 / (values + delta))
        CP = np.power(1 / np.exp(8), coeff)

        return CP

    def get_bin_idx(self, chr, pos):
        sizes = np.load(self.cfg.hic_path + self.cfg.sizes_file, allow_pickle=True).item()
        chr = ['chr' + str(x - 1) for x in chr]
        chr_start = [sizes[key] for key in chr]

        return pos + chr_start

    def _prepare(self):

        for molecule in self.data:
            node_features = molecule['atom_type'].long()

            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list

            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features

            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features

            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class HiCDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, cfg, chr):
        t0 = time.time()

        self.cfg = cfg
        self.chr = chr
        self.num_atom_type = self.cfg.genome_len
        self.num_bond_type = self.cfg.cp_resolution

        if self.cfg.dataset == 'HiC_Rao_10kb':
            self.train = MoleculeDGL(self.cfg, self.chr, 'train')
            # self.val = MoleculeDGL(data_dir, 'val', num_graphs=24445)
            # self.test = MoleculeDGL(data_dir, 'test', num_graphs=5000)
        print("Time taken: {:.4f}s".format(time.time() - t0))


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def make_full_graph(g):
    """
        Converting the given graph to fully connected
        This function just makes full connections
        removes available edge features 
    """

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['feat'] = g.ndata['feat']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()

    try:
        full_g.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
    except:
        pass

    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except:
        pass

    return full_g


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    return g


def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1

    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g


class HiCDataset(torch.utils.data.Dataset):

    def __init__(self, name, cfg, chr):
        """
            Loading HiC dataset
        """
        start = time.time()
        print("Loading Chromosome %s from dataset %s..." % (str(chr), name))
        self.chr = chr
        self.name = name
        self.cell = cfg.cell
        self.cfg = cfg
        # self.dataset = self.get_data()

        data_dir = 'data/HiC/'

        with open(data_dir + name + '.pkl', "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))

        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(labels, dtype=torch.float64)
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _make_full_graph(self):
        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]

    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

    def _add_wl_positional_encodings(self):
        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.train.graph_lists = [wl_positional_encoding(g) for g in self.train.graph_lists]
        self.val.graph_lists = [wl_positional_encoding(g) for g in self.val.graph_lists]
        self.test.graph_lists = [wl_positional_encoding(g) for g in self.test.graph_lists]


if __name__ == '__main__':
    cfg = Config()

    chr = 21
    HiC_data_ob = HiCDatasetDGL(cfg, chr)

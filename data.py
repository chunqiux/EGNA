import torch
from torch.utils.data import Dataset
import pickle
from typing import Tuple


class BindingData(Dataset):
    """ Preprocess data for DataLoader """
    def __init__(self, prot_feat_path: str, lig_info_path: str, dist_info_path: str):
        """
        :param prot_feat_path: Path of the raw protein features
        :param lig_info_path: Path of the raw vertex features and adjacency matrices of ligands
        :param dist_info_path: Path of the distance matrices of the complex
        """
        # Load raw features for preprocessing
        with open(dist_info_path, "rb") as dif:
            self.dist_info = pickle.load(dif)
        with open(lig_info_path, "rb") as lif:
            self.lig_info = pickle.load(lif)
        with open(prot_feat_path, "rb") as pff:
            self.res_feat = pickle.load(pff)

        self.input_list = self.combine_input()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index) -> Tuple:
        return self.input_list[index]

    def combine_input(self):
        """
        Combine all raw features
        :return: A list of combined input features
        """
        assert self.dist_info["pkt_scaf_res_idx"].shape[0] > 0, \
            "No residue in the protein is identified in the binding pocket! "

        input_list = []
        adj_mats = self.calc_adjacency_matrix()
        comb_input = (
            adj_mats[0],
            adj_mats[1],
            adj_mats[2],
            self.assign_node_attr(),
            self.lig_info["atom_feat"],
            self.dist_info["pkt_scaf_res_idx"],
            adj_mats[0].shape[0],
            adj_mats[1].shape[1]
        )
        input_list.append(comb_input)

        return input_list

    def calc_adjacency_matrix(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the adjacency matrices of the protein and ligand
        :return: A tuple contains the adjacency matrix of the scaffold region, binding pocket and ligand
        """
        omega, epsilon = 4., 2.
        scaf_dist_mat = self.dist_info["pp_dist"]
        pkt_dist_mat = self.dist_info["pl_dist"]

        assert torch.min(pkt_dist_mat) > 0, f"The minimum distance between two residues is {torch.min(pkt_dist_mat)}"

        scaf_adj_mat = torch.div(omega, scaf_dist_mat.clamp(min=epsilon))
        pkt_adj_mat = torch.div(omega, pkt_dist_mat.clamp(min=epsilon))
        lig_adj_mat = self.lig_info["edge_idx"].to_dense()

        return scaf_adj_mat, pkt_adj_mat, lig_adj_mat

    def assign_node_attr(self) -> torch.Tensor:
        """
        Extract the scaffold region from the whole protein
        :return: The raw residue features of the scaffold region
        """
        return self.res_feat[self.dist_info["scaf_res_idx"], :]


def data_collate(batch: list) -> list:
    """
    Combine multiple samples into one for convenience.
    """
    batch_pp_mat = torch.block_diag(*[b[0] for b in batch])
    batch_pl_mat = torch.block_diag(*[b[1] for b in batch])
    batch_ll_mat = torch.block_diag(*[b[2] for b in batch])
    batch_prot_node_feat = torch.cat([b[3] for b in batch], dim=0)
    batch_lig_node_feat = torch.cat([b[4] for b in batch], dim=0)

    total_len = 0
    batch_res_idx = []
    prot_graph_idx, lig_graph_idx = [], []
    for i, b in enumerate(batch):
        batch_res_idx.append(torch.from_numpy(b[5] + total_len))
        total_len += b[6]
        prot_graph_idx.append(torch.zeros(len(b[5]), dtype=torch.int64) + i)
        lig_graph_idx.append(torch.zeros(b[7], dtype=torch.int64) + i)
    batch_res_idx = torch.cat(batch_res_idx, dim=0)
    prot_graph_idx = torch.cat(prot_graph_idx, dim=0)
    lig_graph_idx = torch.cat(lig_graph_idx, dim=0)

    batch_data = [batch_prot_node_feat, batch_lig_node_feat, batch_pp_mat, batch_pl_mat, batch_ll_mat,
                  batch_res_idx, prot_graph_idx, lig_graph_idx]

    return batch_data

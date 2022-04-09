import numpy as np
import torch
import pickle
import os
import shutil
from Bio.PDB import *
from Bio.SeqUtils import seq1
from ligand_feature import Ligand
from run_hhblits import HHblits


TMP_DATA_DIR = "tmp_raw_features"


class RawFeatureExtraction(object):
    """
    Extract raw vertex features and adjacency matrices.
    """
    def __init__(self, hhblits_path: str, db_path: str,
                 prot_path: str, lig_path: str, lig_format: str, n_cpu: int):
        """
        :param prot_path: The path of the protein structure file
        :param lig_path: The path of the ligand file
        :param lig_format: The file format (sdf or mol2) of the ligand file
        """
        self.hhblits_path = hhblits_path
        self.db_path = db_path
        self.prot_path = prot_path
        self.lig_path = lig_path
        self.n_cpu = n_cpu

        # Only support sdf or mol2
        if lig_format == "sdf":
            self.suffix = "sdf"
        elif lig_format == "mol2":
            self.suffix = "mol2"
        else:
            raise RuntimeError(f"The format '{lig_format}' of the ligand file is not supported!")

        # Create temporary directory for input data & raw features
        if not os.path.exists(TMP_DATA_DIR):
            os.mkdir(TMP_DATA_DIR)
        self.tmp_prot_path = os.path.join(TMP_DATA_DIR, "protein.pdb")
        self.tmp_lig_path = os.path.join(TMP_DATA_DIR, "ligand." + self.suffix)
        shutil.copy(prot_path, self.tmp_prot_path)
        shutil.copy(lig_path, self.tmp_lig_path)

        # define paths of output raw features
        self.seq_file_path = f"{TMP_DATA_DIR}/protein.fasta"
        self.coord_file_path = f"{TMP_DATA_DIR}/coord.pkl"
        self.ligand_info_path = f"{TMP_DATA_DIR}/ligand_info.pkl"
        self.dist_info_path = f"{TMP_DATA_DIR}/dist_info.pkl"
        self.res_feature_path = f"{TMP_DATA_DIR}/res_feat.pkl"

        self.hhm_dir = f"{TMP_DATA_DIR}/tmp_hhms"
        if not os.path.exists(self.hhm_dir):
            os.mkdir(self.hhm_dir)

    def extract_sequences(self):
        """ Extract sequences from all PDB files and write it to a fasta file"""
        fasta_list = []
        s = PDBParser().get_structure('protein', self.tmp_prot_path)
        for chain in s[0]:
            seq = []
            het_flag = True
            for res in chain:
                res_name = seq1(res.get_resname())
                # Only 20 standard AAs are considered
                if res.get_id()[0] != ' ':
                    continue
                if res_name != 'X':
                    het_flag = False
                seq.append(res_name)
            if het_flag:
                continue
            seq = "".join(seq)
            fasta_list.append(">{}:{}\n{}\n".format("Protein", chain.get_id(), seq))
            print(fasta_list[-1])
        # Save
        with open(self.seq_file_path, 'w') as out_f:
            out_f.writelines(fasta_list)

    def get_protein_ha_coordinates(self):
        """ Save coordinates of heavy atoms"""
        s = PDBParser().get_structure('protein', self.tmp_prot_path)
        heavy_atom_coords, heavy_atom_types = [], []
        ca_indices, cb_indices = [], []
        chain_end_indices, res_end_indices = [], []
        idx = 0
        for chain in s[0]:
            het_flag = True
            for res in chain:
                res_name = seq1(res.get_resname())
                # Only 20 standard AAs are considered
                if res.get_id()[0] != ' ':
                    continue
                if res_name != 'X':
                    het_flag = False
                exist_ca, exist_cb = False, False
                n_ha = 0
                for atom in res:
                    if atom.get_altloc() != ' ' and atom.get_altloc() != 'A':
                        continue
                    atom_name = atom.get_name()
                    if atom_name[0] == 'H':
                        continue
                    heavy_atom_coords.append(atom.coord)
                    heavy_atom_types.append(atom.element)
                    if atom_name[:2] == "CA":
                        exist_ca = True
                        ca_indices.append(idx)
                    elif atom_name[:2] == "CB":
                        exist_cb = True
                        cb_indices.append(idx)
                    idx += 1
                    n_ha += 1
                assert n_ha > 0, f"The number of heavy atoms in chain {chain.get_id()} {res.get_id()} is zero!"
                if not exist_ca:
                    ca_indices.append(len(heavy_atom_coords[:-n_ha]))
                if not exist_cb:
                    cb_indices.append(ca_indices[-1])
                res_end_indices.append(idx)
            if het_flag:
                continue
            chain_end_indices.append(idx)

        coord_dict = {
            "ha_coords": np.array(heavy_atom_coords),
            "ha_types": heavy_atom_types,
            "ch_end_idxs": chain_end_indices,
            "res_end_idx": res_end_indices,
            "ca_idx": ca_indices,
            "cb_idx": cb_indices
        }
        # Save
        with open(self.coord_file_path, "wb") as of:
            pickle.dump(coord_dict, of)

    def get_ligand_info(self):
        """Get raw ligand vertex features and adjacency matrices"""
        lig = Ligand(self.tmp_lig_path, self.suffix)
        atomic_features = lig.get_atomic_features()
        edge_index = lig.get_atomic_graph()
        atomic_coords = lig.get_coords()
        lig_dict = {
            "atom_feat": atomic_features,
            "atom_coords": atomic_coords,
            "edge_idx": edge_index
        }
        # Save
        with open(self.ligand_info_path, "wb") as lif:
            pickle.dump(lig_dict, lif)

    def get_distance_matrix(self, scaffold_th: float = 20, pocket_th: float = 10):
        """
        Get the distance matrix of the protein and the complex
        :param scaffold_th: The distance threshold to determine the scaffold region
        :param pocket_th: The distance threshold to determine the binding pocket
        """
        # load the protein/ligand atomic coordinate file
        with open(self.coord_file_path, "rb") as cf:
            prot_coords = pickle.load(cf)
        with open(self.ligand_info_path, "rb") as lif:
            lig_info = pickle.load(lif)

        cb_coords = np.array(prot_coords["ha_coords"][prot_coords["cb_idx"]])
        lig_coords = lig_info["atom_coords"]

        lp_dist = calc_euclidean_distance_matrix(cb_coords, lig_coords)
        scaf_res_idx = get_nonzero_col_index(lp_dist, scaffold_th)
        scaf_lp_dist = lp_dist[:, scaf_res_idx]
        # get the residue index of the pocket in the scaffold
        pkt_scaf_res_idx = get_nonzero_col_index(scaf_lp_dist, pocket_th)
        # get the residue index of the pocket in the whole protein
        pkt_res_idx = get_nonzero_col_index(lp_dist, pocket_th)
        pkt_lp_dist = lp_dist[:, pkt_res_idx]
        # get the intra-residue distance matrix
        scaf_pp_dist = calc_euclidean_distance_matrix(cb_coords[scaf_res_idx, :], cb_coords[scaf_res_idx, :])

        dist_dict = {
            "pl_dist": torch.tensor(pkt_lp_dist).t(),
            "pp_dist": torch.tensor(scaf_pp_dist),
            "scaf_res_idx": scaf_res_idx,
            "pkt_res_idx": pkt_res_idx,
            "pkt_scaf_res_idx": pkt_scaf_res_idx
        }

        with open(self.dist_info_path, "wb") as dif:
            pickle.dump(dist_dict, dif)

    def get_residue_features(self):
        """ Get raw protein vertex features """
        s = PDBParser().get_structure("protein", self.tmp_prot_path)
        res_feat_list = []
        for chain in s[0]:
            # Only chains of AAs are considered
            het_flag = True
            for res in chain:
                res_name = seq1(res.get_resname())
                # Only 20 standard AAs are considered
                if res.get_id()[0] != ' ':
                    continue
                if res_name != 'X':
                    het_flag = False
            if het_flag:
                continue

            if chain.get_id() == ' ':
                hhm_path = f"{self.hhm_dir}/Protein_.hhm"
            else:
                hhm_path = f"{self.hhm_dir}/Protein_{chain.get_id()}.hhm"

            hhblits = HHblits(self.hhblits_path, self.db_path, self.n_cpu)
            hhblits.run_hhblits(self.seq_file_path, self.hhm_dir)
            res_feat_list.append(get_hhm(hhm_path))

        hhm_tensor = torch.cat(res_feat_list, dim=0)
        # Save
        with open(self.res_feature_path, "wb") as rff:
            pickle.dump(hhm_tensor, rff)


def get_hhm(hhm_path: str) -> torch.Tensor:
    """
    Featurize sequence profiles
    :param hhm_path: The path of the sequence profile
    :return: The raw residue features for a protein sequence
    """
    mat = []
    with open(hhm_path, 'r') as mat_file:
        for line in mat_file:
            if line.strip() == '#':
                break
        for i in range(4):
            mat_file.readline()
        for i, line in enumerate(mat_file):
            if i % 3 == 0:
                line = line.strip().split()
                row = list(map(lambda x: 2 ** (-0.001 * float(x)) if x != '*' else 0.0, line[2:-1]))
            elif i % 3 == 1:
                line = line.strip().split('\t')
                row.extend(list(map(lambda x: 2 ** (-0.001 * float(x)) if x != '*' else 0.0, line[:-3])))
                row.extend(list(map(lambda x: float(x) * 0.001 / 20, line[-3:])))
            else:
                mat.append(row)
    return torch.tensor(mat)


def calc_euclidean_distance_matrix(xyz_1: np.ndarray, xyz_2: np.ndarray) -> np.ndarray:
    """
    Calculate the distance matrix between two point sets.
    :param xyz_1: The Cartesian coordinates of the point set 1
    :param xyz_2:The Cartesian coordinates of the point set 2
    :return: 2D distance matrix
    """
    xyz_2 = np.expand_dims(xyz_2, axis=1)
    dist_mat = np.sqrt(np.sum(np.square(xyz_1 - xyz_2), axis=2))
    return dist_mat


def get_nonzero_col_index(dist_mat: np.ndarray, th: float) -> np.ndarray:
    """
    Get the index of columns, each element of which is smaller than th, of the  distance matrix.
    :param dist_mat: 2d Distance matrix
    :param th: threshold (unit is Angstrom)
    :return: 1d array of column index
    """
    return np.where(dist_mat > th, 0, dist_mat).sum(axis=0).nonzero()[0]

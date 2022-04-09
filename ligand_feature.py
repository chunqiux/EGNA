import openbabel.openbabel as ob
from openbabel import pybel
import torch

class Ligand(object):
    """ Featurize ligand properties """
    def __init__(self, lig_path: str, lig_format: str):
        """
        :param lig_path: The path of the ligand file
        :param lig_format: The format of the ligand file
        """
        obc = ob.OBConversion()
        obc.SetInFormat(lig_format)
        self.mol = ob.OBMol()
        if not obc.ReadFile(self.mol, lig_path):
            raise RuntimeError("Reading ligand file is failed.")
        self.mol.DeleteHydrogens()
        self.coords = []

    def get_coords(self) -> torch.Tensor:
        """
        Get the Cartesian coordinates of all heavy atoms in the ligand
        :return: The torch tensor of the coordinates
        """
        mol = pybel.Molecule(self.mol)
        if len(self.coords) == 0:
            self.coords = torch.tensor([a.coords for a in mol], dtype=torch.float)
        return self.coords

    def get_atomic_features(self) -> torch.Tensor:
        """
        Get the atomic features
        :return: The torch tensor of the atomic features
        """
        atomic_features = []
        for atom in ob.OBMolAtomIter(self.mol):
            descriptor = [
                atom.GetAtomicNum() == 5, atom.GetAtomicNum() == 6, atom.GetAtomicNum() == 7,  # B, C, N
                atom.GetAtomicNum() == 8, atom.GetAtomicNum() == 15, atom.GetAtomicNum() == 16,  # O, P, S
                atom.GetAtomicNum() == 34,  # Se
                atom.GetAtomicNum() in {9, 17, 35, 53, 85}, atom.IsMetal(),  # halogen, metal
                atom.IsAromatic(), atom.IsInRing(),
                # atom.IsHbondDonor() returns zero and thus has no effect
                atom.IsHbondDonor(), atom.IsHbondAcceptor(),
                atom.GetHyb(), atom.GetPartialCharge()
            ]
            atomic_features.append([float(d) for d in descriptor])
        return torch.tensor(atomic_features, dtype=torch.float)

    def get_atomic_graph(self) -> torch.Tensor:
        """
        Construct the graph of the ligand
        :return: The sparse adjacency matrix
        """
        n_atoms = len(self.get_coords())
        edge_index, edge_value = [], []
        for bond in ob.OBMolBondIter(self.mol):
            i = bond.GetBeginAtomIdx() - 1
            j = bond.GetEndAtomIdx() - 1
            edge_index.extend([(i, j), (j, i)])
            edge_value.extend([1.0, 1.0])

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = torch.sparse_coo_tensor(edge_index.t(), edge_value, size=(n_atoms, n_atoms))
        return edge_index

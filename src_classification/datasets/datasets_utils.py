
import torch
import numpy as np
from rdkit import Chem
from collections import defaultdict
from torch_geometric.data import Data
# from rdkit.Chem import Descriptors
# from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


# note this is different from the 2D case
allowable_features = {
    # atom maps in {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9; 'P': 15, 'S': 16, 'CL': 17}
    # "possible_atomic_num_list": [1, 6, 7, 8, 9, 15, 16, 17, "unknown"],
    'possible_atomic_num_list':       list(range(1, 119)),
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE,
    ],
}

# shorten the sentence
feats = allowable_features


def mol_to_graph_data_obj_simple_3D(mol, add_full_edge_list=False):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric. Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr"""
    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    atom_count = defaultdict(int)
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atom_count[atomic_number] += 1
        atomic_number = 19
        if atomic_number not in feats["possible_atomic_num_list"]:
            atomic_number = "unknown"

        atom_feature = \
            [feats["possible_atomic_num_list"].index(atomic_number)] + \
            [feats["possible_chirality_list"].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    N = len(mol.GetAtoms())

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = \
                [feats["possible_bonds"].index(bond.GetBondType())] + \
                [feats["possible_bond_dirs"].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)


        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

        if add_full_edge_list:
            full_edge_list = []
            for i in range(N):
                for j in range(i + 1, N):
                    full_edge_list.append((i, j))
                    full_edge_list.append((j, i))
            full_edge_index = torch.tensor(np.array(full_edge_list).T, dtype=torch.long)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

        if add_full_edge_list:
            full_edge_index = torch.empty((2, 0), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    if add_full_edge_list:
        data = Data(
            x=x,
            edge_index=edge_index,
            full_edge_index=full_edge_index,
            edge_attr=edge_attr,
            positions=positions,
        )
    else:
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            positions=positions,
        )
    return data, atom_count

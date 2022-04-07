import os
from itertools import repeat

import numpy as np
import torch
from descriptastorus.descriptors import rdDescriptors
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


RDKIT_PROPS = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
               'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
               'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
               'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
               'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
               'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
               'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
               'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
               'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
               'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
               'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
               'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
               'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
               'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
               'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
               'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
               'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']


def rdkit_functional_group_label_features_generator(smiles):
    """
    Generates functional group label for a molecule using RDKit.
    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D features.
    """
    # smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    # if type(mol) != str else mol
    generator = rdDescriptors.RDKit2D(RDKIT_PROPS)
    features = generator.process(smiles)[1:]
    features = np.array(features)
    features[features != 0] = 1
    return features


class MoleculeMotifDataset(InMemoryDataset):
    def __init__(self, root, dataset,
                 transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.dataset = dataset
        self.root = root

        super(MoleculeMotifDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

        self.motif_file = os.path.join(root, 'processed', 'motif.pt')
        self.process_motif_file()
        self.motif_label_list = torch.load(self.motif_file)

        print('Dataset: {}\nData: {}\nMotif: {}'.format(self.dataset, self.data, self.motif_label_list.size()))

    def process_motif_file(self):
        if not os.path.exists(self.motif_file):
            smiles_file = os.path.join(self.root, 'processed', 'smiles.csv')
            data_smiles_list = []
            with open(smiles_file, 'r') as f:
                lines = f.readlines()
            for smiles in lines:
                data_smiles_list.append(smiles.strip())

            motif_label_list = []
            for smiles in tqdm(data_smiles_list):
                label = rdkit_functional_group_label_features_generator(smiles)
                motif_label_list.append(label)

            self.motif_label_list = torch.LongTensor(motif_label_list)
            torch.save(self.motif_label_list, self.motif_file)
        return

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        data.y = self.motif_label_list[idx]
        return data

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        return

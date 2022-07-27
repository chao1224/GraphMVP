import json
from collections import OrderedDict
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles



if __name__ == '__main__':
    # from DeepDTA data
    dataset = 'kiba'

    print('convert data from DeepDTA for ', dataset)
    fpath = dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
    drugs = []
    prots = []

    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train', 'test']

    print('{} drugs\t{} unique drugs'.format(len(drugs), len(set(drugs))))

    with open('smiles.csv', 'w') as f:
        print('smiles', file=f)
        for drug in drugs:
            print(drug, file=f)
    with open('protein.csv', 'w') as f:
        print('protein', file=f)
        for prot in prots:
            print(prot, file=f)

    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]
        elif opt == 'test':
            rows, cols = rows[valid_fold], cols[valid_fold]
        # with open(opt + '.csv', 'w') as f:
        #     f.write('compound_iso_smiles,target_sequence,affinity\n')
        #     for pair_ind in range(len(rows)):
        #         ls = []
        #         ls += [drugs[rows[pair_ind]]]
        #         ls += [prots[cols[pair_ind]]]
        #         ls += [affinity[rows[pair_ind], cols[pair_ind]]]
        #         f.write(','.join(map(str, ls)) + '\n')
        with open(opt + '.csv', 'w') as f:
            f.write('smiles_id,target_id,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [rows[pair_ind]]
                ls += [cols[pair_ind]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')

    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))

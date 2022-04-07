import pickle
from collections import Counter
from multiprocessing import Pool

import tqdm

BOND_FEATURES = ['BondType', 'BondDir']


def atom_to_vocab(mol, atom):
    """
    Convert atom to vocabulary. The convention is based on atom type and bond type.
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated atom vocabulary with its contexts.
    """
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])

    # The generated atom_vocab is too long?
    return output


def bond_to_vocab(mol, bond):
    """
    Convert bond to vocabulary. The convention is based on atom type and bond type.
    Considering one-hop neighbor atoms
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated bond vocabulary with its contexts.
    """
    nei = Counter()
    two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
    two_indices = [a.GetIdx() for a in two_neighbors]
    for nei_atom in two_neighbors:
        for a in nei_atom.GetNeighbors():
            a_idx = a.GetIdx()
            if a_idx in two_indices:
                continue
            tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
            nei[str(nei_atom.GetSymbol()) + '-' + get_bond_feature_name(tmp_bond)] += 1
    keys = list(nei.keys())
    keys.sort()
    output = get_bond_feature_name(bond)
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output


def get_bond_feature_name(bond):
    """
    Return the string format of bond features.
    Bond features are surrounded with ()
    """
    ret = []
    for bond_feature in BOND_FEATURES:
        fea = eval(f"bond.Get{bond_feature}")()
        ret.append(str(fea))

    return '(' + '-'.join(ret) + ')'


class TorchVocab(object):
    def __init__(self, counter, max_size=None, min_freq=1, specials=('<pad>', '<other>'), vocab_type='atom'):
        """
        :param counter:
        :param max_size:
        :param min_freq:
        :param specials:
        :param vocab_type: 'atom': atom atom_vocab; 'bond': bond atom_vocab.
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        if vocab_type in ('atom', 'bond'):
            self.vocab_type = vocab_type
        else:
            raise ValueError('Wrong input for vocab_type!')
        self.itos = list(specials)

        max_size = None if max_size is None else max_size + len(self.itos)
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.other_index = 1
        self.pad_index = 0

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
                self.freqs[w] = 0
            self.freqs[w] += v.freqs[w]

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class MolVocab(TorchVocab):
    def __init__(self, molecule_list, max_size=None, min_freq=1, num_workers=1, total_lines=None, vocab_type='atom'):
        if vocab_type in ('atom', 'bond'):
            self.vocab_type = vocab_type
        else:
            raise ValueError('Wrong input for vocab_type!')
        print("Building {} vocab from molecule-list".format((self.vocab_type)))

        from rdkit import RDLogger
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)

        if total_lines is None:
            total_lines = len(molecule_list)

        counter = Counter()
        pbar = tqdm.tqdm(total=total_lines)
        pool = Pool(num_workers)
        res = []
        batch = 50000
        callback = lambda a: pbar.update(batch)
        for i in range(int(total_lines / batch + 1)):
            start = int(batch * i)
            end = min(total_lines, batch * (i + 1))
            res.append(pool.apply_async(MolVocab.read_counter_from_molecules,
                                        args=(molecule_list, start, end, vocab_type,),
                                        callback=callback))
        pool.close()
        pool.join()
        for r in res:
            sub_counter = r.get()
            for k in sub_counter:
                if k not in counter:
                    counter[k] = 0
                counter[k] += sub_counter[k]
        super().__init__(counter, max_size=max_size, min_freq=min_freq, vocab_type=vocab_type)

    @staticmethod
    def read_counter_from_molecules(molecule_list, start, end, vocab_type):
        sub_counter = Counter()
        for i, mol in enumerate(molecule_list):
            if i < start:
                continue
            if i >= end:
                break
            if vocab_type == 'atom':
                for atom in mol.GetAtoms():
                    v = atom_to_vocab(mol, atom)
                    sub_counter[v] += 1
            else:
                for bond in mol.GetBonds():
                    v = bond_to_vocab(mol, bond)
                    sub_counter[v] += 1
        # print("end")
        return sub_counter

    @staticmethod
    def load_vocab(vocab_path: str) -> 'MolVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

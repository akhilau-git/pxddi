from rdkit import Chem
import torch
from torch_geometric.data import Data

ATOM_LIST = ['C','N','O','F','S','Cl','Br','I','P','H']
NUM_ATOM_FEATURES = len(ATOM_LIST) + 3

def atom_features(atom):
    one_hot = [1.0 if atom.GetSymbol()==s else 0.0 for s in ATOM_LIST]
    return one_hot + [atom.GetDegree(), atom.GetFormalCharge(), int(atom.GetIsAromatic())]

def smiles_to_graph(smiles):
    if not isinstance(smiles, str) or not smiles: return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    nf = [atom_features(a) for a in mol.GetAtoms()]
    if not nf: return None
    x = torch.tensor(nf, dtype=torch.float)
    ei = []
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei += [[i,j],[j,i]]
    if not ei: return None
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, smiles=smiles)

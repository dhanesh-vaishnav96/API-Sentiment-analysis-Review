"""inspect_model.py — inspects the GRU model dimensions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import _load_pickle

model = _load_pickle("multi_output_model.pkl")
print("Embedding vocab size:", model.embedding.num_embeddings)
print("Embed dim           :", model.embedding.embedding_dim)
print("Hidden dim          :", model.fc_sent.in_features // 2)
print("Num sent classes    :", model.fc_sent.out_features)
print("Num cat classes     :", model.fc_cat.out_features)

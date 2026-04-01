"""
utils.py — Text cleaning + model/artifact loading utilities.

The 'multi_output_model.pkl' is actually a pickled PyTorch GRUModel instance
trained in Colab (saved as __main__.GRUModel).  We define GRUModel here and
use a custom Unpickler that remaps __main__.GRUModel → utils.GRUModel so
pickle can reconstruct it correctly outside of Colab.
"""

import re
import pickle
import io
import os
import yaml
import nltk
import torch
import torch.nn as nn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn

_vader_analyzer = SentimentIntensityAnalyzer()

# ─── Download NLTK data on first run ──────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "Model")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# ─── GRU Model definition (must match Colab exactly so pickle works) ───────────
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_sent_classes, num_cat_classes):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_sent = nn.Linear(hidden_dim * 2, num_sent_classes)
        self.fc_cat  = nn.Linear(hidden_dim * 2, num_cat_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        sent_out = self.fc_sent(hidden)
        cat_out  = self.fc_cat(hidden)
        return sent_out, cat_out

# ─── Load config ───────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

# ─── Text cleaning (mirrors Colab pipeline) ────────────────────────────────────
# Preserve negation words so "not good" ≠ "good"
_NEGATIONS = {"no", "not", "nor", "neither", "never", "none", "nobody",
              "nothing", "nowhere", "hardly", "barely", "scarcely",
              "doesn't", "don't", "didn't", "won't", "can't", "couldn't",
              "shouldn't", "wouldn't", "isn't", "aren't", "wasn't", "weren't"}

_stop_words = set(stopwords.words("english")) - _NEGATIONS
_lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove URLs
    text = re.sub(r"[^\w\s]", "", text)                    # remove punctuation / emojis
    text = re.sub(r"\d+", "", text)                        # remove numbers
    words = text.split()
    words = [
        _lemmatizer.lemmatize(w)
        for w in words
        if w not in _stop_words
    ]
    return " ".join(words)

# ─── Vocabulary helpers (match Colab logic) ────────────────────────────────────
MAX_LEN = 100

def text_to_sequence(text: str, vocab: dict) -> list:
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]

def pad_sequence(seq: list, max_len: int = MAX_LEN) -> list:
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# ─── Semantic Matching Helpers (Understanding Meaning) ────────────────────────
_ARTIFACT_SYNSET = "artifact.n.01" # All man-made objects
_BILLING_SYNSETS  = {"money.n.01", "payment.n.01", "bill.n.01", "invoice.n.01"}
_SHIPPING_SYNSETS = {"delivery.n.03", "transportation.n.01", "shipping.n.01"}

def has_semantic_ancestor(word: str, target_synsets: set or str) -> bool:
    """Check if word belongs to a semantic group (hypernym check)."""
    if isinstance(target_synsets, str):
        target_synsets = {target_synsets}
        
    synsets = wn.synsets(word)
    for s in synsets:
        for path in s.hypernym_paths():
            for ancestor in path:
                if ancestor.name() in target_synsets:
                    return True
    return False

def adjust_category(text: str, predicted_category: str) -> str:
    """Return category based on semantic understanding of the words."""
    # 1. Tokenize and Tag parts of speech
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    
    # 2. Extract Nouns (objects)
    nouns = [word for word, tag in tagged if tag.startswith('NN')]
    
    # 3. Analyze meaning of nouns
    ship_score    = 0
    billing_score = 0
    product_score = 0

    for n in nouns:
        # Check if it's a physical product object (semantic 'understanding')
        if has_semantic_ancestor(n, _ARTIFACT_SYNSET):
            product_score += 1
            
        # Check for billing/money relation
        if has_semantic_ancestor(n, _BILLING_SYNSETS):
            billing_score += 1
            
        # Check for transportation/delivery relation
        if has_semantic_ancestor(n, _SHIPPING_SYNSETS):
            ship_score += 1
            
    # 4. Keyword Boosters (for common process words)
    text_lower = text.lower()
    if any(k in text_lower for k in ["delivery", "shipping", "courier", "package", "parcel"]):
        ship_score += 2
    if any(k in text_lower for k in ["bill", "price", "refund", "payment", "charge", "tax"]):
        billing_score += 2
    if any(k in text_lower for k in ["product", "quality", "item", "it"]):
        product_score += 0.5 # 'it' is weak but common
        
    scores = {"Shipping": ship_score, "Billing": billing_score, "Product": product_score}
    best_cat, best_score = max(scores.items(), key=lambda x: x[1])

    # 5. Result
    return best_cat if best_score > 0 else predicted_category


def get_vader_sentiment(raw_text: str) -> str:
    """
    Use VADER Sentiment Analyzer to get a robust sentiment prediction.
    VADER inherently understands negations, punctuation, and capitalization.
    """
    score = _vader_analyzer.polarity_scores(raw_text)
    compound = score["compound"]
    
    # Standard thresholding for VADER
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        # Fallback for VADER neutral on obvious technical/process complaints
        text_lower = raw_text.lower()
        if any(w in text_lower for w in ["issue", "problem", "not work", "doesn't work", "broken", "defective", "fail", "bad", "late"]):
            return "Negative"
        return "Neutral"

# ─── Model loading ─────────────────────────────────────────────────────────────
def _load_pickle(filename: str):
    path = os.path.join(MODEL_DIR, filename)

    # Custom Unpickler: remaps __main__.GRUModel → this module's GRUModel
    # so the model saved from Colab (where the class lived in __main__)
    # can be loaded in any other Python context.
    class _FixedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "GRUModel":
                return GRUModel          # always use OUR definition
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return _FixedUnpickler(f).load()

def load_artifacts():
    """
    Load and return:
      tfidf_vectorizer, gru_model, sentiment_encoder, category_encoder, vocab

    About the vocab
    ---------------
    The GRU model was trained in Colab with a vocab of 46 tokens
    (2 special + 44 real words from the 160 training sentences).
    We do NOT have the original training CSV locally, so we cannot
    reconstruct the exact vocab → word-to-index mapping.

    Safe fallback: every word maps to <UNK> (index 1), which is always
    in-range.  For better accuracy, export the vocab dict from Colab
    and load it here from vocab.pkl or vocab.json.
    """
    tfidf             = _load_pickle("tfidf_vectorizer.pkl")
    gru_model         = _load_pickle("multi_output_model.pkl")
    sentiment_encoder = _load_pickle("sentiment_encoder.pkl")
    category_encoder  = _load_pickle("category_encoder.pkl")

    # Minimal vocab — everything unknown maps to index 1 (<UNK>)
    vocab = {"<PAD>": 0, "<UNK>": 1}

    gru_model.eval()
    return tfidf, gru_model, sentiment_encoder, category_encoder, vocab


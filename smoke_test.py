"""
smoke_test.py — verifies the full inference pipeline works.
Run with: python smoke_test.py
"""
import sys
import os

# Ensure utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_artifacts, clean_text, text_to_sequence,
    pad_sequence, adjust_category, MAX_LEN,
)
import torch
import torch.nn.functional as F

print("Loading artifacts...")
tfidf, gru_model, sent_enc, cat_enc, vocab = load_artifacts()
print("Artifacts loaded OK")
print(f"Vocab size: {len(vocab)}")
print(f"Sentiment classes: {sent_enc.classes_}")
print(f"Category classes:  {cat_enc.classes_}")

# Run a test prediction
text    = "The billing was terrible and I never got a refund!"
cleaned = clean_text(text)
print(f"\nInput:   {text}")
print(f"Cleaned: {cleaned}")

seq    = text_to_sequence(cleaned, vocab)
seq    = pad_sequence(seq, MAX_LEN)
tensor = torch.tensor([seq], dtype=torch.long)

with torch.no_grad():
    sent_out, cat_out = gru_model(tensor)

sent_probs = F.softmax(sent_out, dim=1)
cat_probs  = F.softmax(cat_out,  dim=1)

sent_idx  = int(torch.argmax(sent_probs, dim=1).item())
cat_idx   = int(torch.argmax(cat_probs,  dim=1).item())
sentiment = sent_enc.inverse_transform([sent_idx])[0]
raw_cat   = cat_enc.inverse_transform([cat_idx])[0]
category  = adjust_category(text, raw_cat)

print(f"\nSentiment:         {sentiment}  (confidence: {float(torch.max(sent_probs)):.4f})")
print(f"Raw category:      {raw_cat}")
print(f"Adjusted category: {category}")
print("\n✅  ALL OK!")

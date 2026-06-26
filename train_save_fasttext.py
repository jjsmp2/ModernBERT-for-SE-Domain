# train_save_fasttext.py
# Trains FastText on your corpus and saves it
# Run: python train_save_fasttext.py

import os
from pathlib import Path
from gensim.models import FastText as GensimFastText

SE_WORDS = [
    "virus", "class", "requirement", "cloud",
    "elicitation", "maintainability", "container",
    "function", "stack", "repository"
]

# Collect corpus
corpus_lines = []
data_path = Path("results/data/processed")

if data_path.exists():
    for fpath in data_path.rglob("*.txt"):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                corpus_lines.append(f.read().lower().strip())
        except Exception:
            continue

if not corpus_lines:
    print("No processed data found, using fallback corpus...")
    for fpath in Path("results").rglob("*.txt"):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                corpus_lines.append(f.read().lower().strip())
        except Exception:
            continue

print(f"Corpus size: {len(corpus_lines)} documents")

# Train FastText
sentences = [line.split() for line in corpus_lines if line.strip()]
print("Training FastText...")
ft_model = GensimFastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    epochs=10,
    workers=4,
)
print("Training complete.")

# Save model
os.makedirs("results/models", exist_ok=True)
save_path = "results/models/fasttext_se_model"
ft_model.save(save_path)
print(f"Model saved to {save_path}")

# Print Table 5 results
print("\n=== FastText Most Similar Words ===")
for word in SE_WORDS:
    try:
        similar = ft_model.wv.most_similar(word, topn=5)
        words = [w[0] for w in similar]
        print(f"{word}: {', '.join(words)}")
    except Exception as e:
        print(f"{word}: ERROR - {e}")

# GloVe results
print("\n=== GloVe Most Similar Words ===")
try:
    import gensim.downloader as api
    print("Loading GloVe (cached from previous run)...")
    glove = api.load("glove-wiki-gigaword-100")
    for word in SE_WORDS:
        try:
            similar = glove.most_similar(word, topn=5)
            words = [w[0] for w in similar]
            print(f"{word}: {', '.join(words)}")
        except KeyError:
            print(f"{word}: NOT IN VOCAB")
except Exception as e:
    print(f"GloVe error: {e}")
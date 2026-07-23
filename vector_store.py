import os
import glob
import numpy as np
import faiss

print("Loading embedding batches...")

embedding_folder = "../data/embedding_batches"

files = sorted(glob.glob(os.path.join(embedding_folder, "*.npy")))

all_embeddings = []

for file in files:
    emb = np.load(file)
    all_embeddings.append(emb)
    print(f"Loaded {os.path.basename(file)}")

embeddings = np.vstack(all_embeddings)

print("\nTotal embeddings:", embeddings.shape)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

print("Adding embeddings to FAISS...")
index.add(embeddings)

# Save index
faiss.write_index(index, "../data/hate_speech_index.faiss")

print("\nFAISS index created successfully!")
print("Total vectors stored:", index.ntotal)
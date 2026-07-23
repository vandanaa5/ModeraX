import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading cleaned dataset...")

df = pd.read_csv("../data/cleaned_hate_speech.csv")

# Remove missing values
df = df.dropna(subset=["clean_text"])

# Convert to string
df["clean_text"] = df["clean_text"].astype(str)

print("Dataset loaded!")
print("Total Tweets:", len(df))

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")

# Create folder to save embedding batches
output_folder = "../data/embedding_batches"
os.makedirs(output_folder, exist_ok=True)

batch_size = 1000
total = len(df)

print("\nGenerating embeddings...\n")

for start in range(0, total, batch_size):

    end = min(start + batch_size, total)

    texts = df["clean_text"].iloc[start:end].tolist()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    file_name = os.path.join(
        output_folder,
        f"embeddings_{start}_{end}.npy"
    )

    np.save(file_name, embeddings)

    print(f"Saved Batch: {start} - {end}")

print("\nAll embeddings generated successfully!")
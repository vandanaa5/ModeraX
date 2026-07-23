import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

print("Loading FAISS index...")
index = faiss.read_index("../data/hate_speech_index.faiss")

print("Loading dataset...")
df = pd.read_csv("../data/cleaned_hate_speech.csv")
df = df.dropna(subset=["clean_text"])
df["clean_text"] = df["clean_text"].astype(str)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=5):
    # Convert query to embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        results.append({
            "tweet": df.iloc[idx]["clean_text"],
            "label": int(df.iloc[idx]["Label"])
        })

    return results

if __name__ == "__main__":
    query = input("Enter your query: ")

    results = retrieve(query)

    print("\nTop 5 Similar Tweets:\n")

    for i, item in enumerate(results, 1):
        label = "Hate Speech" if item["label"] == 1 else "Non-Hate Speech"

        print(f"{i}. Tweet : {item['tweet']}")
        print(f"   Label : {label}\n")
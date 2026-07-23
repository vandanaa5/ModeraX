import faiss

index = faiss.read_index("../data/hate_speech_index.faiss")

print("Total vectors:", index.ntotal)
print("Vector dimension:", index.d)
import pandas as pd
from pymongo import MongoClient

print("Loading cleaned dataset...")

df = pd.read_csv("../data/cleaned_hate_speech.csv")

df = df.dropna(subset=["clean_text"])
df["clean_text"] = df["clean_text"].astype(str)

print("Dataset loaded!")

print("Connecting to MongoDB...")

client = MongoClient(
    "mongodb://localhost:27017/",
    serverSelectionTimeoutMS=5000
)

# Test the connection
client.admin.command("ping")

print("Connected successfully!")

db = client["hate_speech_db"]
collection = db["tweets"]

print("Clearing old documents...")
collection.delete_many({})

records = []

for _, row in df.iterrows():
    records.append({
        "tweet": row["clean_text"],
        "label": int(row["Label"])
    })

print("Inserting documents...")

collection.insert_many(records)

print("Documents inserted:", collection.count_documents({}))
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk

# Download tokenizer (only the first time)
nltk.download("punkt")

# Load dataset
df = pd.read_csv("../data/HateSpeechDatasetBalanced.csv")

# Remove duplicate tweets
df = df.drop_duplicates(subset="Content")

# Function to clean text
def clean_text(text):
    text = str(text).lower()                 # Convert to lowercase
    text = re.sub(r"http\S+", "", text)      # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip() # Remove extra spaces
    return text

# Apply cleaning
df["clean_text"] = df["Content"].apply(clean_text)

# Tokenize
df["tokens"] = df["clean_text"].apply(word_tokenize)

# Display sample
print(df[["Content", "clean_text", "tokens"]].head())

# Save cleaned dataset
df.to_csv("../data/cleaned_hate_speech.csv", index=False)

print("\n Cleaned dataset saved successfully!")
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from ingest import ingest_documents

# Step 1: Load chunks
chunks = ingest_documents()
print(f"✅ Loaded {len(chunks)} chunks for embedding.")

# Step 2: Prepare texts and metadata
texts = [chunk["text"] for chunk in chunks]
metadatas = [{"doc_name": chunk["doc_name"], "chunk_id": chunk["chunk_id"]} for chunk in chunks]

# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# Step 4: Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 5: Save everything
with open("vector_store.pkl", "wb") as f:
    pickle.dump((index, texts, metadatas), f)

print("✅ Vector store built and saved as vector_store.pkl")

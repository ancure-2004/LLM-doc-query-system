import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Step 1: Load the vector store
with open("vector_store.pkl", "rb") as f:
    index, texts, metadatas = pickle.load(f)

# Step 2: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Define a simple retriever function
def retrieve_top_k(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = []
    for i in indices[0]:
        result = {
            "text": texts[i],
            "metadata": metadatas[i]
        }
        results.append(result)
    return results

# Step 4: Ask a question
query = input("❓ Enter your question: ")
results = retrieve_top_k(query)

print("\n🔍 Top Results:")
for i, r in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"📄 Chunk: {r['text']}")
    print(f"🧾 Metadata: {r['metadata']}")


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# 1. Connect to ChromaDB
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embeddings
)


# ============================================================
# 2. Check Total Chunks
# ============================================================

total_chunks = db._collection.count()

print("=" * 60)
print("CHROMADB INFORMATION")
print("=" * 60)

print("Total chunks:", total_chunks)


# ============================================================
# Files stored in ChromaDB
# ============================================================

data = db.get()

sources = sorted(
    set(
        metadata["source"]
        for metadata in data["metadatas"]
        if metadata and "source" in metadata
    )
)

print("\n" + "=" * 60)
print("FILES STORED IN CHROMADB")
print("=" * 60)

print("Total unique files:", len(sources))

for i, source in enumerate(sources, 1):
    print(f"{i}. {source}")

# ============================================================
# show chunks per file
# ============================================================

from collections import Counter

data = db.get()

file_counts = Counter(
    metadata["source"]
    for metadata in data["metadatas"]
    if metadata and "source" in metadata
)

print("\n" + "=" * 60)
print("FILES AND CHUNK COUNTS")
print("=" * 60)

print("Total chunks:", len(data["ids"]))
print("Total unique files:", len(file_counts))

for source, count in sorted(file_counts.items()):
    print(f"{source:<30} {count} chunks")




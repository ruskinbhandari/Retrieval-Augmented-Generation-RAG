import chromadb

client = chromadb.PersistentClient(
    path="db/chroma_db"
)

collection = client.get_collection("langchain")

old_sources = [
    "docs/Google.txt",
    "docs/Microsoft.txt",
    "docs/Nvidia.txt",
    "docs/Spacex.txt",
    "docs/Tesla.txt",
    "docs/Worldlink.txt",
    "docs/attention-is-all-you-need.pdf",
]

data = collection.get(include=["metadatas"])

ids_to_delete = []

for doc_id, metadata in zip(data["ids"], data["metadatas"]):

    source = metadata.get("source", "")

    if source in old_sources:
        ids_to_delete.append(doc_id)

print(f"Found {len(ids_to_delete)} old chunks")

if ids_to_delete:
    collection.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} old chunks")
else:
    print("No old chunks found")

print(f"Remaining chunks: {collection.count()}")

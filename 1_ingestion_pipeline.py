import os
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Configuration
# ============================================================

DOCS_PATH = "docs"
PERSISTENT_DIRECTORY = "db/chroma_db"


# ============================================================
# Embedding Model
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 16
    }
)


# ============================================================
# Connect to Existing ChromaDB
# ============================================================

db = Chroma(
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=embeddings
)

print("=" * 60)
print("EXISTING CHROMADB")
print("=" * 60)

existing_count = db._collection.count()

print("Existing chunks:", existing_count)


# ============================================================
# Find Existing Sources
# ============================================================

data = db.get()

existing_sources = set(
    metadata["source"]
    for metadata in data["metadatas"]
    if metadata and "source" in metadata
)

print("\nExisting files:")

for source in sorted(existing_sources):
    print(" -", source)


# ============================================================
# Load New Documents
# ============================================================

print("\n" + "=" * 60)
print("CHECKING DOCUMENTS")
print("=" * 60)

documents_to_add = []


# ============================================================
# TXT + Markdown / MkDocs
# ============================================================

for extension in ["txt", "md"]:

    loader = DirectoryLoader(
        DOCS_PATH,
        glob=f"**/*.{extension}",
        loader_cls=TextLoader
    )

    documents = loader.load()

    for doc in documents:

        source = doc.metadata.get("source")

        if source in existing_sources:

            print(f"SKIP: {source}")

        else:

            print(f"NEW: {source}")

            documents_to_add.append(doc)


# ============================================================
# PDF Files
# ============================================================

pdf_files = []

for root, dirs, files in os.walk(DOCS_PATH):

    for filename in files:

        if filename.lower().endswith(".pdf"):

            pdf_files.append(
                os.path.join(root, filename)
            )


for pdf_path in pdf_files:

    source = pdf_path

    if source in existing_sources:

        print(f"SKIP: {source}")

    else:

        print(f"NEW PDF: {source}")

        loader = UnstructuredPDFLoader(
            pdf_path,
            strategy="fast"
        )

        pdf_documents = loader.load()

        documents_to_add.extend(pdf_documents)


# ============================================================
# Nothing New?
# ============================================================

if not documents_to_add:

    print("\nNo new documents found.")
    print("ChromaDB is already up to date.")
    exit()


print("\nNew documents:", len(documents_to_add))


# ============================================================
# Split New Documents
# ============================================================

print("\n" + "=" * 60)
print("SPLITTING NEW DOCUMENTS")
print("=" * 60)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(
    documents_to_add
)

print("New chunks:", len(chunks))


# ============================================================
# Add New Chunks to Chroma
# ============================================================

print("\n" + "=" * 60)
print("ADDING NEW CHUNKS TO CHROMADB")
print("=" * 60)

batch_size = 32

for start in range(0, len(chunks), batch_size):

    end = min(
        start + batch_size,
        len(chunks)
    )

    batch = chunks[start:end]

    print(
        f"Adding chunks "
        f"{start + 1}-{end} "
        f"of {len(chunks)}..."
    )

    db.add_documents(batch)


# ============================================================
# Final Check
# ============================================================

final_count = db._collection.count()

print("\n" + "=" * 60)
print("INGESTION COMPLETE")
print("=" * 60)

print("Previous chunks:", existing_count)
print("New chunks:", len(chunks))
print("Total chunks:", final_count)





# import os
# from langchain_community.document_loaders import TextLoader, DirectoryLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from dotenv import load_dotenv


# load_dotenv()


# def load_documents(docs_path="docs"):
#     """Load all text files from the docs directory."""

#     print(f"Loading documents from {docs_path}...")

#     if not os.path.exists(docs_path):
#         raise FileNotFoundError(
#             f"The directory {docs_path} does not exist. "
#             f"Please create it and add your documents."
#         )

#     loader = DirectoryLoader(
#         path=docs_path,
#         glob="**/*.txt",
#         loader_cls=TextLoader
#     )

#     documents = loader.load()

#     if len(documents) == 0:
#         raise FileNotFoundError(
#             f"No .txt files found in {docs_path}. "
#             f"Please add your documents."
#         )

#     print(f"Found {len(documents)} documents.")

#     # Show first 2 documents
#     for i, doc in enumerate(documents[:2]):
#         print(f"\nDocument {i + 1}:")
#         print(f"  Source: {doc.metadata['source']}")
#         print(f"  Content length: {len(doc.page_content)} characters")
#         print(f"  Content preview: {doc.page_content[:100]}...")
#         print(f"  Metadata: {doc.metadata}")

#     return documents


# def split_documents(documents, chunk_size=1000, chunk_overlap=0):
#     """Split documents into smaller chunks."""

#     print("\nSplitting documents into chunks...")

#     text_splitter = CharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )

#     chunks = text_splitter.split_documents(documents)

#     print(f"Total chunks created: {len(chunks)}")

#     # Show first 5 chunks
#     for i, chunk in enumerate(chunks[:5]):
#         print(f"\n--- Chunk {i + 1} ---")
#         print(f"Source: {chunk.metadata['source']}")
#         print(f"Length: {len(chunk.page_content)} characters")
#         print("Content:")
#         print(chunk.page_content)
#         print("-" * 50)

#     if len(chunks) > 5:
#         print(f"\n... and {len(chunks) - 5} more chunks")

#     return chunks


# def create_vector_store(chunks, persist_directory="db/chroma_db"):
#     """Create ChromaDB vector store and add documents in batches."""

#     print("\nCreating embeddings and storing in ChromaDB...")

#     embedding_model = HuggingFaceEmbeddings(
#         model_name="BAAI/bge-small-en-v1.5",
#         model_kwargs={
#             "device": "cpu"
#         },
#         encode_kwargs={
#             "normalize_embeddings": True,
#             "batch_size": 16
#         }
#     )

#     print("--- Creating vector store ---")

#     vectorstore = Chroma(
#         persist_directory=persist_directory,
#         embedding_function=embedding_model,
#         collection_metadata={
#             "hnsw:space": "cosine"
#         }
#     )

#     batch_size = 32

#     total_chunks = len(chunks)

#     for start in range(0, total_chunks, batch_size):

#         end = min(start + batch_size, total_chunks)

#         batch = chunks[start:end]

#         print(
#             f"Embedding chunks "
#             f"{start + 1}-{end} of {total_chunks}..."
#         )

#         vectorstore.add_documents(batch)

#     print("\n--- Finished creating vector store ---")

#     count = vectorstore._collection.count()

#     print(f"Vector store created and saved to {persist_directory}")
#     print(f"Documents stored in Chroma: {count}")

#     return vectorstore


# def main():
#     """Main RAG ingestion pipeline."""

#     print("=== RAG Document Ingestion Pipeline ===\n")

#     docs_path = "docs"
#     persistent_directory = "db/chroma_db"

#     # --------------------------------------------------
#     # Check existing Chroma database
#     # --------------------------------------------------

#     if os.path.exists(persistent_directory):

#         print("Checking existing Chroma database...")

#         # Temporary embedding model for connecting to Chroma
#         embedding_model = HuggingFaceEmbeddings(
#             model_name="BAAI/bge-small-en-v1.5",
#             model_kwargs={
#                 "device": "cpu"
#             },
#             encode_kwargs={
#                 "normalize_embeddings": True
#             }
#         )

#         vectorstore = Chroma(
#             persist_directory=persistent_directory,
#             embedding_function=embedding_model
#         )

#         count = vectorstore._collection.count()

#         print(f"Existing documents in Chroma: {count}")

#         if count > 0:
#             print("\n✅ Vector store already contains documents.")
#             print("Skipping ingestion.")
#             return vectorstore

#         print("\n⚠️ Chroma directory exists but contains 0 documents.")
#         print("Re-processing documents...\n")

#     # --------------------------------------------------
#     # Step 1: Load documents
#     # --------------------------------------------------

#     documents = load_documents(docs_path)

#     # --------------------------------------------------
#     # Step 2: Split documents
#     # --------------------------------------------------

#     chunks = split_documents(
#         documents,
#         chunk_size=1000,
#         chunk_overlap=0
#     )

#     # --------------------------------------------------
#     # Step 3: Create embeddings and Chroma database
#     # --------------------------------------------------

#     vectorstore = create_vector_store(
#         chunks,
#         persistent_directory
#     )

#     # --------------------------------------------------
#     # Finished
#     # --------------------------------------------------

#     print("\n" + "=" * 50)
#     print("✅ INGESTION COMPLETE!")
#     print("=" * 50)

#     print(
#         f"Documents stored in Chroma: "
#         f"{vectorstore._collection.count()}"
#     )

#     return vectorstore


# if __name__ == "__main__":
#     main()









# List of LangChain Document: 
# documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]
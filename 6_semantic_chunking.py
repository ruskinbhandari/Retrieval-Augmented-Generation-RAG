from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load all .txt files dynamically from docs/
loader = DirectoryLoader(
    "docs",
    glob="**/*.txt",
    loader_cls=TextLoader
)

documents = loader.load()

# Semantic Chunker - groups by meaning, not structure
semantic_splitter = SemanticChunker(
    embeddings=HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    ),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)

# Split dynamic documents
chunks = semantic_splitter.split_documents(documents)

print("SEMANTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: ({len(chunk.page_content)} chars)")
    print(f'"{chunk.page_content}"')
    print(f"Source: {chunk.metadata.get('source')}")
    print()
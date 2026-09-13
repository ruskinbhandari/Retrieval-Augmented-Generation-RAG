from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

DOCS_DIR = Path("docs")


# ============================================================
# LOAD DOCUMENTS DYNAMICALLY
# ============================================================

def load_documents(docs_dir):

    documents = []

    for file_path in docs_dir.rglob("*"):

        # Ignore directories
        if not file_path.is_file():
            continue

        file_extension = file_path.suffix.lower()

        try:

            # ------------------------------------------------
            # TXT
            # ------------------------------------------------

            if file_extension == ".txt":

                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )

            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            elif file_extension == ".pdf":

                loader = PyPDFLoader(
                    str(file_path)
                )

            # ------------------------------------------------
            # MARKDOWN / MKDOCS
            # ------------------------------------------------

            elif file_extension in [".md", ".markdown"]:

                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )

            # ------------------------------------------------
            # UNSUPPORTED FILE
            # ------------------------------------------------

            else:
                print(f"Skipping: {file_path}")
                continue

            # Load file
            loaded_documents = loader.load()

            # Add our own metadata
            for document in loaded_documents:

                document.metadata["file_type"] = file_extension

                document.metadata["file_name"] = file_path.name

                document.metadata["folder"] = file_path.parent.name

                document.metadata["relative_path"] = str(
                    file_path.relative_to(docs_dir)
                )

            documents.extend(loaded_documents)

            print(f"Loaded: {file_path}")

        except Exception as e:

            print(f"ERROR: {file_path}")
            print(f"       {e}")

    return documents


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

print("\n" + "=" * 60)
print("DYNAMIC DOCUMENT LOADING")
print("=" * 60)

documents = load_documents(DOCS_DIR)

print(
    f"\nTotal documents/pages loaded: "
    f"{len(documents)}"
)


# ============================================================
# RECURSIVE CHARACTER TEXT SPLITTER
# ============================================================

print("\n" + "=" * 60)
print("RECURSIVE CHARACTER TEXT SPLITTER")
print("=" * 60)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
    chunk_size=1000,
    chunk_overlap=150
)


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

chunks2 = recursive_splitter.split_documents(
    documents
)

print(
    f"Total chunks: "
    f"{len(chunks2)}"
)


# ============================================================
# DISPLAY FIRST 10 CHUNKS
# ============================================================

for i, chunk in enumerate(chunks2[:10], 1):

    print("\n" + "-" * 60)

    print(
        f"Chunk {i}: "
        f"({len(chunk.page_content)} chars)"
    )

    print(
        f"Source: "
        f"{chunk.metadata.get('source')}"
    )

    print(
        f"File: "
        f"{chunk.metadata.get('file_name')}"
    )

    print(
        f"Type: "
        f"{chunk.metadata.get('file_type')}"
    )

    print(
        f"Folder: "
        f"{chunk.metadata.get('folder')}"
    )

    print(
        f"Path: "
        f"{chunk.metadata.get('relative_path')}"
    )

    print(
        f"\nContent:\n"
        f"{chunk.page_content}"
    )



# from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# # tesla_text = """Tesla's Q3 Results

# # Tesla reported record revenue of $25.2B in Q3 2024.

# # Model Y Performance

# # The Model Y became the best-selling vehicle globally, with 350,000 units sold.

# # Production Challenges

# # Supply chain issues caused a 12% increase in production costs.

# # This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""


# # splitter1 = CharacterTextSplitter(
# #     separator=" ",  # Default separator. Other options include ["\n\n", "\n", ". ", " ", ""]
# #     chunk_size=100,
# #     chunk_overlap=0
# # )

# # chunks1 = splitter1.split_text(tesla_text)
# # for i, chunk in enumerate(chunks1, 1):
# #     print(f"Chunk {i}: ({len(chunk)} chars)")
# #     print(f'"{chunk}"')
# #     print()


# # from langchain_text_splitters import RecursiveCharacterTextSplitter


# #TEST
# # text_splitter = RecursiveCharacterTextSplitter(
# #     chunk_size=1000,
# #     chunk_overlap=100
# # )

# # chunks = text_splitter.split_documents(documents)


# # Example 2: RecursiveCharacterTextSplitter fixes this
# print("\n" + "=" * 60)
# print("2. RECURSIVE CHARACTER TEXT SPLITTER SOLUTION")
# print("=" * 60)

# chunks2 = recursive_splitter.split_documents(documents)
#     separators=["\n\n", "\n", ". ", " ", ""],  # Multiple separators
#     chunk_size=100,
#     chunk_overlap=0

# chunks2 = recursive_splitter.split_text(tesla_text)
# print(f"Same problem text, but with RecursiveCharacterTextSplitter:")
# for i, chunk in enumerate(chunks2, 1):
#     print(f"Chunk {i}: ({len(chunk)} chars)")
#     print(f'"{chunk}"')
#     print()
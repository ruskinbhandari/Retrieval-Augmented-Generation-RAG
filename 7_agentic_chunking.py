from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# Load documents dynamically
loader = DirectoryLoader(
    "docs",
    glob="**/*.txt",
    loader_cls=TextLoader
)

documents = loader.load()

# Process each document
for document in documents:

    prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{document.page_content}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

    print("🤖 Asking AI to chunk the document...")

    response = llm.invoke(prompt)
    marked_text = response.content

    # Split the text at the markers
    chunks = marked_text.split("<<<SPLIT>>>")

    # Clean up the chunks
    clean_chunks = []

    for chunk in chunks:
        cleaned = chunk.strip()

        if cleaned:
            clean_chunks.append(cleaned)

    # Show results
    print("\n🎯 AGENTIC CHUNKING RESULTS:")
    print("=" * 50)

    for i, chunk in enumerate(clean_chunks, 1):
        print(f"Chunk {i}: ({len(chunk)} chars)")
        print(f'"{chunk}"')
        print()
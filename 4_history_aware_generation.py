# import time
# from pathlib import Path
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# # ============================================================
# # Configuration
# # ============================================================

# load_dotenv()

# BASE_DIR = Path(__file__).resolve().parent
# PERSISTENT_DIRECTORY = str(BASE_DIR / "db" / "chroma_db")

# # Number of documents to retrieve
# TOP_K = 3

# # Maximum characters taken from each retrieved document
# MAX_CHARS_PER_DOC = 1800

# # Keep only the most recent conversation messages
# # 4 messages = 2 user questions + 2 answers
# MAX_HISTORY_MESSAGES = 4


# # ============================================================
# # Embeddings
# # ============================================================

# print("Loading embedding model...")

# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5",
#     model_kwargs={
#         "device": "cpu"
#     },
#     encode_kwargs={
#         "normalize_embeddings": True
#     }
# )


# # ============================================================
# # ChromaDB
# # ============================================================

# db = Chroma(
#     persist_directory=PERSISTENT_DIRECTORY,
#     embedding_function=embeddings
# )

# print(f"Chroma documents: {db._collection.count()}")


# # ============================================================
# # Ollama LLM
# # ============================================================

# print("Loading Ollama model...")

# model = ChatOllama(
#     model="qwen2.5:3b",
#     temperature=0,

#     # Limit generated answer length.
#     # This can significantly reduce CPU generation time.
#     num_predict=300,

#     # Keep context reasonable for your laptop.
#     num_ctx=4096,

#     # One CPU-based model is generally enough for this setup.
#     num_thread=8,
# )


# # ============================================================
# # Conversation history
# # ============================================================

# chat_history = []


# # ============================================================
# # Retrieve documents
# # ============================================================

# def retrieve_documents(question):
#     """
#     Retrieve the most relevant documents from ChromaDB.
#     """

#     start_time = time.perf_counter()

#     results = db.similarity_search_with_score(
#         question,
#         k=TOP_K
#     )

#     elapsed = time.perf_counter() - start_time

#     print(f"\nRetrieval time: {elapsed:.2f} seconds")

#     documents = []

#     for i, (doc, score) in enumerate(results, 1):

#         source = doc.metadata.get("source", "Unknown")

#         content = doc.page_content.strip()

#         # Prevent huge chunks from being sent to Qwen.
#         if len(content) > MAX_CHARS_PER_DOC:
#             content = content[:MAX_CHARS_PER_DOC] + "..."

#         print(f"\n  Document {i}")
#         print(f"  Source: {source}")
#         print(f"  Distance: {score:.4f}")
#         print(f"  Content: {preview}...")

#         documents.append(
#             {
#                 "source": source,
#                 "content": content,
#             }
#         )

#     return documents


# # ============================================================
# # Build RAG context
# # ============================================================

# def build_context(documents):

#     context_parts = []

#     for i, doc in enumerate(documents, 1):

#         context_parts.append(
#             f"""
# Document {i}
# Source: {doc['source']}

# {doc['content']}
# """.strip()
#         )

#     return "\n\n---\n\n".join(context_parts)


# # ============================================================
# # Ask question
# # ============================================================

# def ask_question(user_question):

#     print(f"\n{'=' * 60}")
#     print(f"Question: {user_question}")
#     print(f"{'=' * 60}")

#     total_start = time.perf_counter()

#     # --------------------------------------------------------
#     # IMPORTANT SPEED OPTIMIZATION
#     # --------------------------------------------------------
#     #
#     # We DO NOT use a separate LLM call to rewrite the question.
#     #
#     # Previously:
#     #
#     # User question
#     #       ↓
#     # Qwen rewrites question
#     #       ↓
#     # Chroma retrieval
#     #       ↓
#     # Qwen generates answer
#     #
#     # Now:
#     #
#     # User question
#     #       ↓
#     # Chroma retrieval
#     #       ↓
#     # Qwen generates answer
#     #
#     # This removes one complete Qwen inference call.
#     # --------------------------------------------------------

#     search_question = user_question

#     # --------------------------------------------------------
#     # Retrieve documents
#     # --------------------------------------------------------

#     documents = retrieve_documents(search_question)

#     if not documents:

#         print("\nNo relevant documents found.")

#         answer = (
#             "I don't have enough information to answer that "
#             "question based on the provided documents."
#         )

#         print(f"\nAnswer: {answer}")

#         return answer

#     context = build_context(documents)

#     # --------------------------------------------------------
#     # Keep only recent history
#     # --------------------------------------------------------

#     recent_history = chat_history[-MAX_HISTORY_MESSAGES:]

#     # --------------------------------------------------------
#     # RAG prompt
#     # --------------------------------------------------------

#     system_prompt = """
# You are a helpful RAG assistant.

# Answer the user's question using ONLY the information
# contained in the provided documents.

# Important rules:

# 1. Do not invent information.
# 2. Do not use outside knowledge.
# 3. If the documents do not contain the answer, say:
#    "I don't have enough information to answer that question
#    based on the provided documents."
# 4. Be concise and direct.
# 5. If the question refers to previous conversation,
#    use the recent conversation history together with the
#    retrieved documents.
# 6. Prefer exact information from the documents over assumptions.
# """

#     user_prompt = f"""
# Retrieved documents:

# {context}

# User question:
# {user_question}

# Answer the question based only on the retrieved documents.
# """

#     messages = [
#         SystemMessage(content=system_prompt)
#     ]

#     # Add recent conversation history.
#     messages.extend(recent_history)

#     # Add current question.
#     messages.append(
#         HumanMessage(content=user_prompt)
#     )

#     # --------------------------------------------------------
#     # Generate answer
#     # --------------------------------------------------------

#     llm_start = time.perf_counter()

#     result = model.invoke(messages)

#     llm_elapsed = time.perf_counter() - llm_start

#     answer = result.content.strip()

#     # --------------------------------------------------------
#     # Save conversation history
#     # --------------------------------------------------------

#     chat_history.append(
#         HumanMessage(content=user_question)
#     )

#     chat_history.append(
#         AIMessage(content=answer)
#     )

#     # Keep memory small.
#     if len(chat_history) > MAX_HISTORY_MESSAGES:
#         del chat_history[:-MAX_HISTORY_MESSAGES]

#     # --------------------------------------------------------
#     # Timing
#     # --------------------------------------------------------

#     total_elapsed = time.perf_counter() - total_start

#     print(f"\nLLM generation time: {llm_elapsed:.2f} seconds")
#     print(f"Total question time: {total_elapsed:.2f} seconds")

#     # --------------------------------------------------------
#     # Answer
#     # --------------------------------------------------------

#     print(f"\nAnswer:\n{answer}")

#     # --------------------------------------------------------
#     # Sources
#     # --------------------------------------------------------

#     print("\nSources:")

#     unique_sources = []

#     for doc in documents:

#         source = doc["source"]

#         if source not in unique_sources:
#             unique_sources.append(source)

#     for source in unique_sources:
#         print(f"  - {source}")

#     return answer


# # ============================================================
# # Start chat
# # ============================================================

# def start_chat():

#     print("\n" + "=" * 60)
#     print("Local RAG Chat")
#     print("=" * 60)

#     print(f"Chroma DB : {PERSISTENT_DIRECTORY}")
#     print(f"Top K     : {TOP_K}")
#     print(f"History   : {MAX_HISTORY_MESSAGES} messages")
#     print(f"Max chars : {MAX_CHARS_PER_DOC} per document")

#     print("\nAsk questions about your documents.")
#     print("Type 'quit' or 'exit' to stop.")
#     print("Type 'clear' to clear conversation history.")

#     while True:

#         try:

#             question = input("\nYou: ").strip()

#         except KeyboardInterrupt:

#             print("\n\nGoodbye!")
#             break

#         except EOFError:

#             print("\n\nGoodbye!")
#             break

#         if not question:
#             continue

#         # ----------------------------------------------------
#         # Exit
#         # ----------------------------------------------------

#         if question.lower() in {"quit", "exit"}:

#             print("Goodbye!")
#             break

#         # ----------------------------------------------------
#         # Clear history
#         # ----------------------------------------------------

#         if question.lower() == "clear":

#             chat_history.clear()

#             print("Conversation history cleared.")

#             continue

#         # ----------------------------------------------------
#         # Ask
#         # ----------------------------------------------------

#         ask_question(question)


# # ============================================================
# # Main
# # ============================================================

# if __name__ == "__main__":
#     start_chat()












from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pathlib import Path

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = str(
    Path(__file__).resolve().parent / "db" / "chroma_db"
)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
    
)
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
print("Total documents in Chroma:", db._collection.count())

# Set up AI model
model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        lines = doc.page_content.split("\n")[:2]
    preview = "\n".join(lines)

    print(f"\n  Doc {i}:")
    print(f"    Source: {source}")
    print(f"    Content: {preview}...")
    
    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ").strip()

        if not question:
            print("Please enter a question.")
            continue
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()
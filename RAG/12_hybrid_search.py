from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma 
from langchain_core.documents import Document 
from dotenv import load_dotenv 
from langchain_core.messages import HumanMessage, SystemMessage
import os 

load_dotenv("./../.env")

gemini_api_key = os.getenv("GOOGLE_API_KEY")

chunks = [
    "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
    "Tesla Cybertruck production ramp begins in 2024.",
    "Google is a large technology company with global operations.",
    "Tesla reported strong quarterly results. Tesla continues to lead in electric vehicles. Tesla announced new manufacturing facilities.",
    "SpaceX develops Starship rockets for Mars missions.",
    "The tech giant acquired the code repository platform for software development.",
    "NVIDIA designs Starship architecture for their new GPUs.",
    "Tesla Tesla Tesla financial quarterly results improved significantly.",
    "Cybertruck reservations exceeded company expectations.",
    "Microsoft is a large technology company with global operations.", 
    "Apple announced new iPhone features for developers.",
    "The apple orchard harvest was excellent this year.",
    "Python programming language is widely used in AI.",
    "The python snake can grow up to 20 feet long.",
    "Java coffee beans are imported from Indonesia.", 
    "Java programming requires understanding of object-oriented concepts.",
    "Orange juice sales increased during winter months.",
    "Orange County reported new housing developments."
]


documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]

print("Sample Data:")
for i, chunk in enumerate(chunks, 1):
    print(f"{i}. {chunk}")

print("\n" + "="*80)

print("Setting up Vector Retriever.....")

embedding_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore=Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k":2})

# Test semantic search
test_query = "space exploration company" #works in vector search but wouldn't work with keyword search

print(f"Testing: '{test_query}'")
test_docs = vector_retriever.invoke(test_query)
for doc in test_docs:
    print(f"Found: {doc.page_content}")

# 2. BM25 Retriever (Keyword Search)
print("Setting up BM25 Retriever...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3


# Test exact keyword matching
# test_query = "space exploration company"
test_query = "Cybertruck"
# test_query = "Tesla"

print(f"Testing: '{test_query}'")
test_docs = bm25_retriever.invoke(test_query)
for doc in test_docs:
    print(f"Found: {doc.page_content}")

#  3. Hybrid Retriever (Combination)
print("Setting up Hybrid Retriever...")
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # Equal weight to vector and keyword search
)

print("Setup complete!\n")

# Query 1: Mixed semantic and exact terms

# Vector search understands "purchase cost" semantically
# BM25 search finds exact "7.5 billion" 
# Hybrid should combine both strengths for best result
test_query = "purchase cost 7.5 billion"

retrieved_chunks = hybrid_retriever.invoke(test_query)
for i, doc in enumerate(retrieved_chunks, 1):
    print(f"{i}. {doc.page_content}")
print()

print("Query 1 shows how hybrid finds exact financial info using both semantic understanding and keyword matching")


combined_input = f"""Based on the following documents, please answer this question: {test_query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in retrieved_chunks])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

model = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            google_api_key=gemini_api_key
        )

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.text)
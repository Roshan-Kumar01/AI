from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel 
from typing import List 
import os 

load_dotenv("./../.env")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

persistent_directory="db/chroma_db"
embedding_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            google_api_key=gemini_api_key
        )

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

class QueryVariations(BaseModel):
    queries: List[str]


original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "="*60)

retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")
from dotenv import load_dotenv 
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv("./../.env")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

persistent_directory="db/chroma_db"
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db=Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

model = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            google_api_key=gemini_api_key
        )

chat_history=[]

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    if chat_history:

        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        print(f"******************\n message:{messages}\n")

        result = model.invoke(messages)
        search_question = result.text.strip()
        print(f"Searching for: {user_question}")

    else:
        search_question=user_question

    
    retriever = db.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    documents_text = "\n".join(
        f"- {doc.page_content}" for doc in docs
    )
    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {documents_text}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)

    answer=result.text 

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"********************* \n chat_history:{chat_history}\n")
    print(f"Answer: {answer}")
    return answer


def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()
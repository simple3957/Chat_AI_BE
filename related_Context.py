# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# import google.generativeai as genai
import os
# genai.configure(api_key="AIzaSyDm8DdIpiNlNrQGtRQ6V1H7Y9ipkedVz6M")
import warnings
warnings.filterwarnings("ignore")


# Define cache directory
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"


def get_Related_Context(query):
    # Paths
    faiss_index_path = "faiss_index_"


    # Load embedding model from cache
    
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, cache_folder=CACHE_DIR)

    # Load FAISS index (without regenerating)
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.invoke(query)    
        # Combine retrieved texts
    context = "\n".join([doc.page_content for doc in relevant_docs])
    # print("context",context)
    return context


def get_gemini_response(query):
    """Retrieve results from FAISS index and generate a concise response using Gemini AI."""

    faiss_index_path = "faiss_index_"

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, cache_folder=CACHE_DIR)


    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        return "I'm sorry, I couldn't find any relevant information for your query. Could you please provide more details?"


    context = "\n".join([doc.page_content for doc in relevant_docs])

    
    vague_keywords = ["tell me about", "explain", "details", "information"]
    is_vague = any(keyword in query.lower() for keyword in vague_keywords)


    if is_vague:
        prompt = f"""
        You are an AI assistant. Use the following context to provide a brief and general response to the query.
        Avoid going into too much detail or showing the entire content directly.

        Context:
        {context}

        Query:
        {query}
        """
    else:
        prompt = f"""
        You are an AI assistant. Use the following context to answer the query accurately and concisely.
        Avoid showing the entire content directly.

        Context:
        {context}

        Query:
        {query}
        """

    # Use Gemini AI to generate response
    model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
    response = model.generate_content(prompt)

    # Return concise response
    return response.text.strip()
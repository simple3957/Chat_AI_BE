import google.generativeai as genai
import os
from related_Context import get_Related_Context

genai.configure(api_key="AIzaSyDm8DdIpiNlNrQGtRQ6V1H7Y9ipkedVz6M")
import warnings
warnings.filterwarnings("ignore")



def get_gemini_response(query):
    """Retrieve results from FAISS index and generate a response using Gemini AI."""
    # Retrieve relevant documents from FAISS
    context = get_Related_Context(query)

    # Prepare prompt
    prompt = f"""
    You are an AI assistant. Use the following context to answer the query accurately and Dont tell that we are given data like based on the given data.
    
    Context:
    {context}
    
    Query:
    {query}
    """

    # Use Gemini AI to generate response
    model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
    response = model.generate_content(prompt)
    
    return response.text


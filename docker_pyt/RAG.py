import os
import google.generativeai as genai
import chromadb
from pypdf import PdfReader

# API key used and configured. getenv() is used here so GitHub doesn't treat the exposed key as something to tell Google to block
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key = GOOGLE_API_KEY)

# Using PersistentClient to ensure data is stored in chroma_db folder in local computer
client = chromadb.PersistentClient(path = "./chroma_db")
ai_info = client.get_or_create_collection(name = "ai_information", metadata = {"hnsw:space": "cosine"}) # Cosine similarity used for comparisons; creates best, optimal chunks

# Functions for chunking text; splits text into 250-word chunks with 50-word overlap. The overlap is solely for avoiding grabbing parts of words instead of whole words
def chunkify(text, chunk_size = 250, overlap = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

# Function for database population; processes PDFs, creates embeddings, and saves them to chroma_db
def populate_database(file_list):
    for file_path in file_list:
        print(f"Indexing {file_path}...")
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                chunks = chunkify(text)
                documents, metadatas, ids, embeddings = [], [], [], []
                
                for i, chunk in enumerate(chunks):
                    result = genai.embed_content(model = "models/gemini-embedding-001",content = chunk,task_type = "retrieval_document")
                    
                    documents.append(chunk)
                    embeddings.append(result["embedding"])

                    metadatas.append({"source": file_path, "page": page_num + 1})
                    ids.append(f"{file_path}_p{page_num}_c{i}")
                
                ai_info.add(ids = ids, documents = documents, metadatas = metadatas, embeddings = embeddings)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# FUnction for retrieving answer; asks question, acquires top 5 chunks, gives proper context to the AI, and then a response is generated
def get_answer(question):
    query_result = genai.embed_content(model = "models/gemini-embedding-001", content = question, task_type = "retrieval_query")
    
    results = ai_info.query(query_embeddings = [query_result["embedding"]], n_results = 5)
    
    context_text = "\n\n".join(results['documents'][0])
    sources = results['metadatas'][0]
    
    prompt = f"""
    Please use the provided information to answer. 
    If you cannot answer it with certainty, say you do not know.

    CONTEXT:
    {context_text}

    QUESTION:
    {question}
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
    except:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

    return response.text, sources

# Main; starts up and continues AI interaction
if __name__ == "__main__":
    if ai_info.count() == 0:
        print("Initial setup: Indexing documents...")
        pdf_files = ["docker_pyt/data/The Book of Python.pdf", "docker_pyt/data/EnC++clopedia.pdf", "docker_pyt/data/New World EnC++clopedia.pdf"]
        populate_database(pdf_files)
        print(f"Indexing complete. {ai_info.count()} chunks stored.")

    print("\nThis is a Python and C++ chatbot! Come chat with me!")
    print("If you'd like to quit, just type 'exit'!")
    
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        answer, citations = get_answer(query)
        
        print(f"\nAI Response: {answer}")
        
        print("\nSources referenced:")
        unique_sources = set()
        for doc in citations:
            unique_sources.add(f"- {doc['source']} (Page {doc['page']})")
        
        for s in sorted(unique_sources):
            print(s)
import google.generativeai as genai
import chromadb
from pypdf import PdfReader

GOOGLE_API_KEY = "AIzaSyD0lmH6ZWp0u3DGMViC6zxAPppgXBJusmY"
genai.configure(GOOGLE_API_KEY)

client = chromadb.PersistentClient(path = "./chroma_db")
ai_info = client.get_or_create_collection(name = "ai_information", metadata = {"hnsw:space": "cosine"})

def chunkify(text, chunk_size = 250, overlap = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

files = ["data/The Book of Python.pdf", "data/EnC++clopedia.pdf", "data/New World EnC++clopedia.pdf"]

for file_path in files:
    print(f"Processing {file_path}...")
    reader = PdfReader(file_path)
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text: continue # Skips empty pages
        
        chunks = chunkify(text)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"Source": file_path, "Page Number": page_num + 1})
            ids.append(f"{file_path}_p{page_num}_c{i}")
        
        ai_info.add(ids = ids, documents = documents, metadatas = metadatas)

print("Vector store successfully populated.")

def embed(chunk):
    model_name = "models/gemini-embedding-001"
    text = chunk
    result = genai.embed_content(model = model_name, content = text, task_type = "retrieval_document")
    return result

def get_answer(question):
    query_vec = genai.embed_content(model = "models/gemini-embedding-001", content = question, task_type="retrieval_query") ["embedding"]

    results = ai_info.query(query_embeddings = [query_vec], n_results = 5)

    context_text = "\n".join(results['documents'][0])
    sources = results['metadatas'][0]

    full_prompt = f"""
    Please use the provided information to answer. 
    If you cannot answer it with certainty, say you do not know.

    CONTEXT:
    {context_text}

    QUESTION:
    {question}
    """

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(full_prompt)

    return response.text, sources

print("Hello! I am an AI who can tell you information about Python and C++ code!")
while True:
    user_q = input("\nAsk me anything! Or type 'quit' to quit: ")
    if user_q.lower() == 'quit': break

    answer, citations = get_answer(user_q)

    print(f"\nAI: {answer}")

    print("\nSources used:")
    for cite in citations:
        print(f"- {cite['source']} (Page {cite.get('page', 'N/A')})")
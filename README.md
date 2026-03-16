# docker_RAG

# Project Overview
This project is a Retrieval-Augmented Generation (RAG) system with Computer Science in mind. Grabbing from my own Python and C++ notes, you can ask it questions, and it will try to give you the best answer based on the notes I took. It will also cite where it got the information from, and it uses many Gemini-related systems to achieve its goal.

# 2. Architectural Choices
To maximize retrieval and system efficiency, I made the following design choices:

   Chunking Strategy: Documents are processed via chunking them 250 words at a time, with a 50-word overlap acting as a way to prevent content from being cut off.

   Vector Database: ChromaDB was selected for its lightweight, persistent storage capabilities. Using PersistentClient() allows the system to save the index locally in the chroma_db/ directory, preventing redundant API calls.

   Similarity Metric: The system uses Cosine Similarity (hnsw:space: cosine). This is ideal for RAG because it measures the orientation (the "meaning") of the text vectors rather than just the Euclidean distance, leading to better matches for conceptual questions.

   LLM & Embeddings: We utilize gemini-embedding-001 for high-dimensional vector representation and gemini-1.5-flash for rapid, cost-effective response generation.

# 3. Key Features
   Source Citations: Every response includes metadata-driven citations (Document Name and Page Num) to provide transparency and provide reassurance to the user.

   Hallucination Guardrail: The system prompt strictly instructs the AI to state "I do not know" if the answer is not present in the retrieved context.

   Persistent Storage: Once indexed, the database remains ready for future sessions without requiring the original PDFs to be re-processed.

# 4. Setup & Installation
Prerequisites
   Python 3.9+ and Conda (miniconda preferably)

Installation
1.  Clone the repository to your local machine.
2.  Install the required dependencies:
    pip install -r requirements.txt
3.  Set the API Key
    Windows (PowerShell): $env:GOOGLE_API_KEY="GOOGLE_API_KEY"
    Mac/Linux: export GOOGLE_API_KEY="GOOGLE_API_KEY"

5. How to Run
   Ensure your PDF documents are located in the data/ folder, then execute the script:
      python RAG.py

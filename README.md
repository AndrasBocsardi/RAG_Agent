# RAG Agent with ReAct Pattern

## Overview
This project is a prototype **Agentic Retrieval-Augmented Generation (RAG) chatbot** built with the **ReAct pattern** using 
The agent demonstrates **autonomous reasoning, tool usage, retrieval integration, and memory** in a conversational setting.  

It loads knowledge from a PDF (*NBA 2024–25 Season Summary*), retrieves relevant sections via a vector database, and generates grounded answers. A math tool is also included to showcase multi-tool usage.

---

## Features
- **Agentic behavior**  
  - Breaks down tasks autonomously and decides when to use tools.  
- **Retrieval-Augmented Generation (RAG)**  
  - PDF ingestion, chunking, embedding, and retrieval via ChromaDB.  
- **ReAct pattern**  
  - Alternates between reasoning and tool execution.  
- **Tools**  
  - `retrieve_documents` → fetches knowledge from the PDF.  
  - `math_calculator` → supports arithmetic operations.  
- **Conversation memory**  
  - Maintains a rolling history of dialogue in a variable (default 20 messages).  
  - Enables multi-turn consistency within a session.  

---

## How It Works

1. **Data Processing**
   - Loads the PDF using `PyPDFLoader`.
   - Splits text into overlapping chunks with `RecursiveCharacterTextSplitter`.

2. **Vector Store**
   - Encodes chunks into embeddings using **Google Gemini embeddings** (`gemini-embedding-001`).
   - Stores them in **ChromaDB** for similarity search.

3. **Reasoning & Acting**
   - User asks a question.
   - Agent decides whether to call retrieval, calculator, or directly answer.
   -  Can combine multiple tools in sequence. Example: retrieve statistical data from the knowledge base, then perform arithmetic calculations on the retrieved values.  

4. **Conversation Memory**
   - Dialogue history is stored in a variable.
   - Provides continuity across multiple turns within a single session.




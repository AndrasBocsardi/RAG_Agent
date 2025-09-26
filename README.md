# RAG Agent with ReAct Pattern

## Overview
This project is a prototype **(RAG) chatbot** built with the **ReAct pattern**.
The agent demonstrates **autonomous reasoning, tool usage, retrieval integration, and memory** in a conversational setting.  

It loads knowledge from a PDF (*NBA 2024–25 Season Summary*), retrieves relevant sections via a vector database, and generates grounded answers. A math tool is also included to showcase multi-tool usage.

---
- The notebook (`.ipynb`) and Python script (`.py`) contain the same code. Both are provided for convenience.  
- To run the Python script, install the dependencies listed in `requirements.txt`.
- To run the notebook, you may need to install `ipykernel` and `jupyter`. 
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
  ![image alt](https://github.com/AndrasBocsardi/RAG_Agent/blob/2fb39ccf5a7cc101f278f82b426ac107f2ee2488/RAG%20example.png)

4. **Conversation Memory**
   - Dialogue history is stored in a variable.
   - Provides continuity across multiple turns within a single session.

---

## Design choices
- **ReAct pattern** 
    - ReAct pattern was chosen to demonstrate agentic behavior, where the model
    can reason about tasks, decide on tool usage, and chain tool calls.
- **Google Gemini API** 
    - Initially tested local HuggingFace models, but due to hardware limitations, they were impractical to run.
    Switched to the free Google Gemini API for both the LLM (gemini-2.5-flash) and embeddings (gemini-embedding-001).

---

## Limitations & Future Work

- **Hardware constraints**  
  Could not use large local models; limited to free-tier cloud APIs.  

- **Conversation memory**  
  Only session-based; no persistence to disk or database.  

- **Knowledge source**  
  Currently only supports a single PDF. Future work could enable multi-document ingestion or external APIs.  

- **Evaluation**  
  No benchmarking included. Latency, accuracy, and scalability analysis could be added.  

- **Interface**  
  Runs in CLI; could be extended to a web interface (ex. Streamlit).  

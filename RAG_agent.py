from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def load_and_process_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load PDF and split it into chunks for processing."""
    pdf_loader = PyPDFLoader(pdf_path)
    
    try:
        pages = pdf_loader.load()
        print(f"PDF has been loaded and has {len(pages)} pages")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pages_split = text_splitter.split_documents(pages)
    
    return pages_split

def setup_vector_store(documents, embeddings, persist_directory: str, collection_name: str):
    """Create ChromaDB vector store and return retriever."""
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    try:
        # Here, we actually create the chroma database using our embeddings model
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print(f"Created ChromaDB vector store!")
        
    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        raise

    # Now we create our retriever with improved search parameters
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    return retriever

# Initialize the system
pdf_path = "NBA_2024_25_Season_Summary.pdf"
persist_directory = os.getcwd()
collection_name = "nba_summary"

# Load and process PDF
pages_split = load_and_process_pdf(pdf_path)

# Setup vector store and retriever
retriever = setup_vector_store(pages_split, embeddings, persist_directory, collection_name)

@tool
def math_calculator(operation: str, a: float, b: float = None):
    """
    Perform basic mathematical operations.
    
    Args:
        operation: The math operation to perform. Options: 'add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'abs', 'round'
        a: First number (required)
        b: Second number (required for binary operations like add, subtract, multiply, divide, power)
    
    Returns:
        String representation of the calculation result
    """
    import math
    
    operation = operation.lower().strip()
    
    try:
        if operation in ['add', '+', 'plus']:
            if b is None:
                return "Error: Addition requires two numbers"
            result = a + b
            return f"{a} + {b} = {result}"
            
        elif operation in ['subtract', '-', 'minus']:
            if b is None:
                return "Error: Subtraction requires two numbers"
            result = a - b
            return f"{a} - {b} = {result}"
            
        elif operation in ['multiply', '*', 'times']:
            if b is None:
                return "Error: Multiplication requires two numbers"
            result = a * b
            return f"{a} × {b} = {result}"
            
        elif operation in ['divide', '/', 'divided by']:
            if b is None:
                return "Error: Division requires two numbers"
            if b == 0:
                return "Error: Division by zero is not allowed"
            result = a / b
            return f"{a} ÷ {b} = {result}"   
        else:
            return f"Error: Unknown operation '{operation}'. Supported operations: add, subtract, multiply, divide, power, sqrt, abs, round, sin, cos, tan"
            
    except Exception as e:
        return f"Error performing calculation: {str(e)}"

@tool
def retrieve_documents(query: str) -> str:
    """Retrieve relevant documents from the vector store based on the query. Use this document to answer questions about the NBA 2024-25 season."""
    
    docs = retriever.invoke(query)
    
    if not docs:
        return f"I found no relevant information for '{query}' in the document. Try rephrasing your question or ask about a different topic."
    
    results = []
    for i, doc in enumerate(docs):   
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

tools = [retrieve_documents, math_calculator] 
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0    


system_prompt = """
You are an AI assistant with access to a PDF document loaded into your knowledge base. 
Your primary responsibility is to answer user questions based on the content of that PDF. 
Always try to ground your responses in the document first. If the information can be found in the PDF, 
provide a clear, accurate, and well-structured answer that summarizes or references the relevant sections. 
If the information is not present in the document, notify the user that the PDF does not cover it, 
and then do your best to answer from your own knowledge. 
You may also use any additional tools that are available to improve the accuracy or completeness of your response. 
Your goal is to prioritize the PDFs content while ensuring the user always receives a helpful answer.
"""

# Create tool registry for better organization
available_tools = {tool.name: tool for tool in tools}

# Conditional edge function
def needs_tool_execution(state: AgentState) -> bool:
    """
    Determine if the last message contains tool calls that need to be executed.
    This controls the flow between reasoning and tool execution in the RAG pipeline.
    """
    last_message = state['messages'][-1]
    return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0 

# LLM Reasoning Agent
def agent_reasoning(state: AgentState) -> AgentState:
    """
    Core reasoning function that processes user input and decides on actions.
    This represents the 'Generate' step in RAG (Retrieval-Augmented Generation).
    """
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(messages)
    return {'messages': [response]}


# Tool Execution Agent  
def execute_tools(state: AgentState) -> AgentState:
    """
    Execute tool calls from the LLM's response.
    This handles both retrieval tools (R in RAG) and other utility tools.
    """
    tool_calls = state['messages'][-1].tool_calls
    print(f"Executing {len(tool_calls)} tool call(s)...")
    
    results = []
    for tool_call in tool_calls:
        print(f"Calling Tool: {tool_call['name']} with args: {tool_call['args']}")
        
        if tool_call['name'] not in available_tools:
            print(f"Error: Tool '{tool_call['name']}' does not exist.")
            result = f"Error: Tool '{tool_call['name']}' is not available. Available tools: {list(available_tools.keys())}"
        else:
            try:
                result = available_tools[tool_call['name']].invoke(tool_call['args'])
                print(f"Tool execution successful. Result length: {len(str(result))} characters")
            except Exception as e:
                result = f"Error executing tool '{tool_call['name']}': {str(e)}"
                print(f"Tool execution failed: {e}")

        # Create tool message with result
        tool_message = ToolMessage(
            tool_call_id=tool_call['id'], 
            name=tool_call['name'], 
            content=str(result)
        )
        results.append(tool_message)

    print("All tools executed. Returning to reasoning agent...")
    return {'messages': results}

# Build the RAG Agent Graph
rag_graph = StateGraph(AgentState)
rag_graph.add_node("reasoning", agent_reasoning)
rag_graph.add_node("tool_execution", execute_tools)

# Define the conditional flow: reasoning -> tool_execution (if needed) -> reasoning -> end
rag_graph.add_conditional_edges(
    "reasoning",
    needs_tool_execution,
    {True: "tool_execution", False: END}
)
rag_graph.add_edge("tool_execution", "reasoning")
rag_graph.set_entry_point("reasoning")

# Compile the RAG agent
rag_agent = rag_graph.compile()


class ConversationMemory:
    """Manages conversation history with configurable memory length."""
    
    def __init__(self, max_messages: int = 20):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep in memory.
                         Default is 20 (about 10 exchanges), which provides good
                         context while staying within token limits.
        """
        self.max_messages = max_messages
        self.conversation_history = []
    
    def add_message(self, message: BaseMessage):
        """Add a message to the conversation history."""
        self.conversation_history.append(message)
        self._trim_history()
    
    def add_messages(self, messages: list[BaseMessage]):
        """Add multiple messages to the conversation history."""
        self.conversation_history.extend(messages)
        self._trim_history()
    
    def get_history(self) -> list[BaseMessage]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear all conversation history."""
        self.conversation_history = []
    
    def _trim_history(self):
        """Keep only the most recent messages within the memory limit."""
        if len(self.conversation_history) > self.max_messages:
            # Keep the most recent messages
            self.conversation_history = self.conversation_history[-self.max_messages:]


def run_rag_agent_with_memory(memory_length: int = 20):
    """
    Main function to run the RAG (Retrieval-Augmented Generation) agent with conversation memory.
    
    This function implements the complete RAG pipeline:
    1. Retrieval: Search relevant documents from the knowledge base
    2. Augmentation: Combine retrieved context with user query  
    3. Generation: Generate response using LLM with retrieved context
    
    Args:
        memory_length: Number of recent messages to remember (default: 20).
                      This equals about 10 exchanges, providing good context
                      while staying within reasonable token limits.
    """
    print("\n" + "="*50)
    print("🤖 RAG AGENT - Retrieval Augmented Generation")
    print("="*50)
    print(f"📚 Knowledge Base: NBA 2024-25 Season Summary")
    print(f"💬 Type 'exit' or 'quit' to end the conversation")
    print("="*50)
    
    # Initialize conversation memory management
    conversation_memory = ConversationMemory(max_messages=memory_length)
    
    while True:
        user_query = input(f"\n💭 Your question: ")
        
        if user_query.lower().strip() in ['exit', 'quit', 'bye']:
            break
            
        # Add user message to conversation memory
        user_message = HumanMessage(content=user_query)
        conversation_memory.add_message(user_message)
        
        # Get full conversation history and invoke RAG agent
        conversation_history = conversation_memory.get_history()
        
        try:
            # Run the RAG pipeline
            result = rag_agent.invoke({"messages": conversation_history})
            
            # Extract and store the final response
            final_response = result['messages'][-1]
            conversation_memory.add_message(final_response)
            
            # Display the response
            print("\n" + "="*50)
            print("🤖 AGENT RESPONSE:")
            print("="*50)
            print(final_response.content)
            
        except Exception as e:
            print(f"\n Error processing your request: {str(e)}")
            print("Please try rephrasing your question or contact support.")


# Start the RAG agent
if __name__ == "__main__":
    run_rag_agent_with_memory()

    
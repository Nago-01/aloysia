import re, os, traceback, warnings, time
from typing import Annotated, Literal, List, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor
from tavily import TavilyClient
import arxiv



# Load environment variables
load_dotenv()

# Supressing warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Defining the state structure for the agent                        
class AgentState(TypedDict):
    """The agent's memory - tracks conversation and context"""
    messages: Annotated[list, add_messages]
    quality_passed: bool
    loop_count: int
    original_query: str
    selected_model: str # 'groq' or 'gemini'


def manage_context_window(messages: list) -> list:
    """
    Middleware to strictly enforce token/context limits.
    1. Truncates all ToolMessages to prevent bloat.
    2. Keeps only last 10 messages (sliding window).
    3. Preserves System Prompt.
    """
    MAX_CURRENT_TOOL_CHARS = 5000
    MAX_HISTORY_TOOL_CHARS = 500
    MAX_TEXT_MSG_CHARS = 2000 
    WINDOW_SIZE = 10
    
    processed = []
    
    # Check if messages is empty
    if not messages:
        return []
    
    # Helper to check if msg is from current turn (simplified logic)
    # We assume the last message in the list is the most recent
    
    # Slice to window size first
    recent_messages = messages[-WINDOW_SIZE:]
    
    for i, msg in enumerate(reversed(recent_messages)):
        # i=0 is the last message (most recent)
        is_current_turn = (i < 2) # Allow last 2 messages to be "current" (tool output + tool call)
        
        if isinstance(msg, ToolMessage):
            limit = MAX_CURRENT_TOOL_CHARS if is_current_turn else MAX_HISTORY_TOOL_CHARS
            if len(msg.content) > limit:
                new_content = msg.content[:limit] + f"\n...[Truncated {len(msg.content)-limit} chars]..."
                msg = ToolMessage(content=new_content, tool_call_id=msg.tool_call_id, name=msg.name)
        elif isinstance(msg, (HumanMessage, AIMessage)) and hasattr(msg, 'content'):
             # Truncate massive text responses (like bibliographies) in history
             if len(msg.content) > MAX_TEXT_MSG_CHARS:
                 new_content = msg.content[:MAX_TEXT_MSG_CHARS] + f"\n...[Truncated {len(msg.content)-MAX_TEXT_MSG_CHARS} chars]..."
                 # Create copy with new content
                 if isinstance(msg, HumanMessage):
                     msg = HumanMessage(content=new_content)
                 else:
                     msg = AIMessage(content=new_content)
        
        processed.insert(0, msg)
        
    # Always keep System Message if it exists in the original list
    sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    if sys_msgs:
        # Check if we already have a system message in processed
        if not any(isinstance(m, SystemMessage) for m in processed):
             processed.insert(0, sys_msgs[0])
    
    # SANITIZATION: Ensure we don't start with a ToolMessage (orphaned)
    # Most APIs require: User -> AI -> Tool -> AI
    # If the first non-System message is a ToolMessage, drop it (and likely its chunks)
    while processed and isinstance(processed[0], (ToolMessage, SystemMessage)):
        if isinstance(processed[0], SystemMessage):
            # Skip system message check here (we handle it above/below)
            if len(processed) > 1 and isinstance(processed[1], ToolMessage):
                 processed.pop(1) # Remove the tool message after system
                 continue
            break
        else:
            processed.pop(0)
            
    # FINAL SANITIZATION: Strict Role Alternation (LangGraph/Groq compatible)
    # Sequence must be: [System] -> Human -> AI -> Tool -> AI -> Human ...
    final_sanitized = []
    
    # Track the last non-system role to enforce alternation
    # Roles: 'human', 'ai', 'tool', 'system'
    last_role = None
    
    for msg in processed:
        # 1. Identify Role
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "human"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, AIMessage):
            role = "ai"
        else:
            continue
            
        # 2. Sequence Logic
        if role == "system":
            if not final_sanitized:
                final_sanitized.append(msg)
            continue # Only one system message at the very top
            
        if not final_sanitized:
            # If no history, we must start with a human message
            if role == "human":
                final_sanitized.append(msg)
                last_role = "human"
            continue
            
        # 3. Handle Alternation Violations
        if role == last_role:
            if role == "human":
                # Consecutive Humans: Merge them or keep latest
                final_sanitized.pop()
            elif role == "ai":
                # Consecutive AI: Keep latest (likely a newer plan or tool call)
                final_sanitized.pop()
            elif role == "tool":
                # Consecutive Tools: This is common (multiple tools called in one turn)
                # We ALLOW consecutive ToolMessages IF they have different IDs
                pass
            
        # 4. Mandatory Precedence: ToolMessage MUST follow an AIMessage with tool_calls
        if role == "tool":
            prev_msg = final_sanitized[-1]
            if not (isinstance(prev_msg, AIMessage) and prev_msg.tool_calls):
                # Orphaned tool message! This triggers 400 error. 
                # We must drop it if we don't have the call.
                continue
        
        final_sanitized.append(msg)
        last_role = role
    
    # FINAL STRUCTURAL LOGGING
    seq = [m.__class__.__name__ for m in final_sanitized]
    print(f"--- Sanitized Sequence: {' -> '.join(seq)} ---")
    
    return final_sanitized
    
    # FINAL STRUCTURAL LOGGING: Help debug 400 Errors
    seq = [m.__class__.__name__ for m in final_sanitized]
    print(f"--- Sanitized Sequence: {' -> '.join(seq)} ---")
    
    return final_sanitized


# Defining tools  
@tool
def parallel_document_analysis(topic: str, doc_list: List[str]) -> str:
    """
    Analyze multiple documents in parallel on a topic.
    """
    from code.rag_init import get_rag

    rag = get_rag()

    def analyze_single_doc(doc_name):
        """Analyze one document"""
        results = rag.db.search(f"{topic} {doc_name}", n_results=3)

        return {
            "doc": doc_name,
            "findings": results["documents"][:3],
            "citations": results["citations"][:3]
        }


    # Run in parallel
    with ThreadPoolExecutor(max_workers=len(doc_list)) as executor:
        analyses = list(executor.map(analyze_single_doc, doc_list))

    synthesis = "# Multi-Document Analysis\n\n"
    for analysis in analyses:
        synthesis += f"## {analysis['doc']}\n"
        for finding in analysis['findings']:
            synthesis += f"- {finding[:200]}...\n"

    return synthesis     


@tool
def rag_search(query: str) -> str:
    """
    Search the internal document knowledge base for information.
    
    **ALWAYS USE THIS FIRST** for any factual questions about provided documents.
    Returns results with source, page number and section information.
    
    Args:
        query: The search query 
        
    Returns:
        Relevant document excerpts with full citations
    """
    from code.rag_init import get_rag
    
    rag = get_rag()
    print("Using cached RAG assistant")


    # Expand short queries for better results
    if len(query.split()) <= 3:
        expanded_query = f"{query} definition meaning explanation overview details"
    else:
        expanded_query = query


    # Get search results with metadata
    results = rag.db.search(expanded_query, n_results=8, use_reranking=True)
    
    
    # Check if we got any results
    if not results["documents"]:
        return "No relevant information found in the documents. You may try web_search for current information."
    
    # Format the results with context
    formatted_results = []
    for i, (doc, metadata, citation, distance) in enumerate(zip(
        results["documents"][:5], 
        results["metadatas"][:5],
        results["citations"][:5], 
        results["distances"][:5]
    ), 1):
        # SANITIZATION: Remove non-printable characters to prevent API errors
        clean_doc = "".join(char for char in doc if char.isprintable() or char in ['\n', '\t'])
        
        formatted_results.append(
            f"**Result {i}** (Relevance: {distance:.3f})\n"
            f"{citation}\n\n"
            f"{clean_doc}\n"
            f"\n"
        )
    
    
    return "\n".join(formatted_results)


@tool
def compare_documents(doc1_name: str, doc2_name: str, topic: str) -> str:
    """
    Compare information from two specific documents on a given topic.
    
    Use this when the user explicitly asks to compare two documents, 
    such as "Compare AMR and Dysmenorrhea papers" or "what's the difference
    between document A and document B on topic Y?"

    Args:
        doc1_name: Name of first document (e.g., "amr.pdf")
        doc2_name: Name of second document (e.g., "dysmenorrhea.pdf")
        topic: The topic/aspect to compare (e.g., "treatment", "causes")

    Returns:
        Side-by-side comaprison of the two documents
    """
    from code.rag_init import get_rag

    rag = get_rag()

    try: 
        # Search for topic in both documents
        query = f"{topic}"
        results = rag.db.search(query, n_results=10, use_reranking=True)

        # Filter results by document
        doc1_results = []
        doc2_results = []

        for doc, metadata, citation in zip(
            results["documents"], 
            results["metadatas"],
            results["citations"]
        ):
            source = metadata.get("source", "").lower()

            if doc1_name.lower() in source:
                doc1_results.append({
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,
                    "citation": citation
                })
            elif doc2_name.lower() in source:
                doc2_results.append({
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,
                    "citation": citation
                })
        

        # Format comparison output
        comparison = f"**COMPARISON: {doc1_name} vs {doc2_name} on '{topic}'**\n\n"

        comparison += f"**{doc1_name}:**\n"
        if doc1_results:
            for i, result in enumerate(doc1_results[:3], 1):
                comparison += f"{i}. {result['citation']}\n{result['content']}\n\n"
        else:
            comparison += f"No relevant information found about '{topic}'\n\n"

        comparison += f"**{doc2_name}:**\n"
        if doc2_results:
            for i, result in enumerate(doc2_results[:3], 1):
                comparison += f"{i}. {result['citation']}\n{result['content']}\n\n"
        else:
            comparison += f"No relevant information found about '{topic}'\n\n"
        
        return comparison
    
    except Exception as e:
        return f"Error comparing documents: {str(e)}"
    

@tool
def generate_bibliography() -> str:
    """
    Generate a formatted bibliography of all documents in the knowledge base.

    Use this when the user asks for:
    - "List of documents"
    - "Show me the bibliography"
    - "What sources do you have?"
    - "Generate a reference list"

    Returns:
        Formatted bibliography with document metadata
    """
    from code.rag_init import get_rag

    rag = get_rag()

    try:
        # Fetch all metadata from Supabase
        all_metadatas = rag.db.list_all_metadata()
        seen_sources = set()
        bibliography = []

        for metadata in all_metadatas:
            source = metadata.get("source")
            if source and source not in seen_sources:
                seen_sources.add(source)
                entry = {
                    "source": source,
                    "title": metadata.get("title", "Unknown Title"),
                    "author": metadata.get("author", "Unknown Author"),
                    "page_count": metadata.get("page_count", "N/A"),
                    "type": metadata.get("type", "N/A"),
                    "date": metadata.get("creation_date", metadata.get("date_from_filename", "N/A"))
                }
                bibliography.append(entry)
        
        # Format bibliography
        output = "BIBLIOGRAPHY\n"
        output += "="*50 + "\n\n"

        for i, entry in enumerate(bibliography, 1):
            output += f"{i}. **{entry['title']}**\n"
            output += f"    Author: {entry['author']}\n"
            output += f"    Source: {entry['source']}\n"
            output += f"    Type: {entry['type'].upper()}\n"
            output += f"    Pages: {entry['page_count']}\n\n"

            if entry['date'] != "N/A":
                output += f"    Date: {entry['date']}\n"

            output += "\n"

        output += "="*50 + "\n"
        output += f"**Total Documents: {len(bibliography)}**\n"

        return output
    except Exception as e:
        return f"Error generating bibliography: {str(e)}"


@tool
def generate_literature_review(topic: str, max_sources: int = 10) -> str:
    """
    Generate a structured academic literature review on a specific topic.

    Use when user asks to:
        - "Write a literature review on X"
        - "Summarize research on X"
        - "What does the literature say about X?"

    Args:
        topic: The research topic
        max_sources: Maximum number of sources (default: 10)

    Returns:
        Formatted literature review with citations
    """
    from code.rag_init import get_rag

    rag = get_rag()

    try:
        results = rag.db.search(topic, n_results=max_sources, use_reranking=True)

        if not results["documents"]:
            return f"No literature found on topic '{topic}'."
        
        # Organize by document source
        docs_content = {}
        for doc, metadata, citation in zip(
            results["documents"], 
            results["metadatas"],
            results["citations"]
        ):
            source = metadata.get("source", "Unknown")
            if source not in docs_content:
                docs_content[source] = {
                    "title": metadata.get("title", source),
                    "author": metadata.get("author", "Unknown Author"),
                    "excerpts": [],
                    "citations": []
                }
            docs_content[source]["excerpts"].append(doc[:500])
            docs_content[source]["citations"].append(citation)

        # Generate structured review
        review = f"# Literature Review: {topic}\n\n"
        review += f"**Documents Analyzed:** {len(docs_content)}\n\n"
        review += "---\n\n"

        # Introduction
        review += "## Overview\n\n"
        review += f"This review synthesizes findings from {len(docs_content)} documents "
        review += f"related to {topic}. The following sections present key findings from each source.\n\n"
        review += "---\n\n"

        # Document-by-document review
        for i, (source, content) in enumerate(docs_content.items(), 1):
            review += f"## {i}. {content['title']}\n\n"
            review += f"**Author:** {content['author']}\n"
            review += f"**Source:** {source}\n\n"
            review += "**Key Findings:**\n\n"
            for j, excerpt in enumerate(content['excerpts'][:3], 1):
                review += f"{j}. {excerpt}...\n\n"
            review += f"**Citations:** {', '.join(set(content['citations'][:2]))}\n\n"
            review += "---\n\n"

        # Synthesis section
        review += "## Synthesis\n\n"
        review += f"Across the {len(docs_content)} reviewed sources, several key themes emerges "
        review += f"regarding {topic}. The literature provides comprehensive insights into "
        review += "various aspects of the topic, offering both theoritical frameworks and practical applications.\n\n"


        # References
        review += "## References\n\n"
        all_citations = []
        for content in docs_content.values():
            all_citations.extend(content['citations'])

        unique_citations = list(set(all_citations))
        for i, citation in enumerate(unique_citations, 1):
            review += f"{i}. {citation}\n"

        return review
    except Exception as e:
        return f"Error generating literature review: {str(e)}"


@tool
def export_bibliography(format: str = "word") -> str:
    """
    Export bibliography in specified format (word, latex, markdown).

    Use when user asks to:
    - "Export bibliography"
    - "Generate LaTex bibliography"
    - "Save references as markdown"

    Args:
        format: Export format - 'word', 'latex', or 'markdown' (default: 'word')

    Returns:
        Path to the exported file
    """
    from code.rag_init import get_rag
    from code.export_utils import (
        generate_latex_bibliography,
        generate_word_bibliography,
        generate_markdown_bibliography
    )

    rag = get_rag()

    try:
        # Fetch all metadata from Supabase
        all_metadatas = rag.db.list_all_metadata()
        seen_sources = set()
        bibliography = []

        for metadata in all_metadatas:
            source = metadata.get("source")
            if source and source not in seen_sources:
                seen_sources.add(source)
                bibliography.append({
                    "source": source,
                    "title": metadata.get("title", "Unknown Title"),
                    "author": metadata.get("author", "Unknown Author"),
                    "page_count": metadata.get("page_count", "N/A"),
                    "type": metadata.get("type", "Unknown"),
                    "date": metadata.get("creation_date", "n.d.")
                })

        
        bibliography.sort(key=lambda x: x["source"])

        if format.lower() == "latex":
            file_path = generate_latex_bibliography(bibliography)
            return f"Bibliography exported to LaTex: {file_path}"
        
        elif format.lower() == "markdown" or format.lower() == "md":
            file_path = generate_markdown_bibliography(bibliography)
            return f"Bibliography exported to Markdown: {file_path}"
        
        else: # Default to word
            file_path = generate_word_bibliography(bibliography)
            return f"Bibliography exported to Word: {file_path}"
    except Exception as e:
        return f"Error exporting bibliography: {str(e)}"
    

@tool
def export_literature_review(topic: str, format: str = "word") -> str:
    """ 
    Generate and export a complete literature review document.
    
    Use when user asks to:
    - "Export literature review on X as Word"
    - "Generate LaTex review on X"
    - "Save review about X as PDF"

    Args:
        topic: The research topic
        format: Export format - 'word', 'latex', or 'markdown' (default: 'word')
    
    Returns:
        Path to the exported document
    """
    from code.rag_init import get_rag
    from code.export_utils import generate_literature_review_document


    rag = get_rag()

    try:
        results = rag.db.search(topic, n_results=10, use_reranking=True)

        if not results.get("documents"):
            return f"No literature found on the topic: {topic}"
        
        sections = []
        docs_content = {}

        for doc, metadata, citation in zip(
            results.get("documents", []),
            results.get("metadatas", []),
            results.get("citations", [])
        ):
            if not isinstance(metadata, dict):
                continue
            source = metadata.get("source", "Unknown")
            if source not in docs_content:
                docs_content[source] = {
                    "source": source,
                    "content": [],
                    "citations": []
                }
            docs_content[source]["content"].append(doc)
            docs_content[source]["citations"].append(citation)

        
        for source, data in docs_content.items():
            combined_content = "\n\n".join(data["content"][:3])
            sections.append({
                "source": data["source"],
                "content": combined_content,
                "citations": list(set(data["citations"]))
            })

        
        file_path = generate_literature_review_document(
            topic=topic,
            sections=sections,
            format=format.lower()
            )
        
        return f"Literature review exported to {format.upper()}: {file_path}"
    
    except Exception as e:
        return f"Error generating literature review: {str(e)}"                           



@tool
def web_search(query: str) -> str:
    """
    Search the web for current, real-time information.
    
    **ONLY USE THIS** when:
    - rag_search returns "No relevant information found"
    - Question is about very recent events (news from last few days)
    - Question needs real-time data (weather, stock prices, current events)
    
    Args:
        query: The search query
        
    Returns:
        Search results from the web
    """
    if not os.getenv("TAVILY_API_KEY"):
        return "Web search is not available. Please set TAVILY_API_KEY in .env file."
    
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        search_tool = tavily_client.search(
            query=query,
            max_results=3,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced"
        )

        results = search_tool["results"]

        if isinstance(results, str):
            return results
        elif isinstance(results, list):
            formatted = []
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    content = result.get("content", result.get("snippet", ""))
                    url = result.get("url", "")
                    formatted.append(f"{i}. {content}\nSource: {url}")
                else:
                    formatted.append(f"{i}. {str(result)}")
            return "\n\n".join(formatted) if formatted else "No results found."
        elif isinstance(results, dict):
            # if it's a single dictionary
            content = results.get("content", results.get("snippet", ""))
            url = results.get("url", "")
            return f"{content}\nSource: {url}"
        else:
            return str(results)
    except Exception as e:
        return f"Error during web search: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.
    Use this when you need to compute numbers, percentages, or do math.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "100 * 0.15", "(500 + 300) / 2")
        
    Returns:
        The calculated result
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def arxiv_search(query: str, max_results: int = 5, save_to_db: bool = False) -> str:
    """
    Search arXiv for academic papers on a topic.
    
    Use when user asks for:
    - "Find papers on X"
    - "What's the latest research on X?"
    - "Search arXiv for X"
    - "Find academic papers about X"
    
    Args:
        query: Search query for arXiv
        max_results: Maximum number of papers to return (default: 5)
        save_to_db: If True, saves papers to local vector DB for future retrieval
        
    Returns:
        Formatted list of papers with titles, authors, abstracts, and arXiv links
    """
    try:
        # Create arXiv client and search
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = list(client.results(search))
        
        if not results:
            return f"No papers found on arXiv for '{query}'."
        
        # Format results
        output = f"**arXiv Search Results for '{query}'**\n"
        output += "=" * 50 + "\n\n"
        
        papers_data = []
        for i, paper in enumerate(results, 1):
            output += f"**{i}. {paper.title}**\n"
            # Robust author parsing: handle unexpected metadata types gracefully
            try:
                author_names = [getattr(a, 'name', str(a)) for a in paper.authors[:3]]
                output += f"   Authors: {', '.join(author_names)}"
            except:
                output += "   Authors: Information unavailable"
            if len(paper.authors) > 3:
                output += f" et al."
            output += "\n"
            output += f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
            output += f"   Abstract: {paper.summary[:300]}...\n"
            output += f"   URL: {paper.entry_id}\n"
            output += f"   PDF: {paper.pdf_url}\n\n"
            
            # Collect data for potential DB storage
            papers_data.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "abstract": paper.summary,
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "published": paper.published.strftime('%Y-%m-%d'),
                "categories": paper.categories
            })
        
        # Optionally save to vector DB
        if save_to_db:
            from code.rag_init import get_rag
            rag = get_rag()
            docs_added = rag.db.add_arxiv_papers(papers_data)
            output += f"\n {docs_added} papers saved to local knowledge base.\n"
        
        return output
        
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"


def get_tools():
    """Return all available tools for the agent"""
    return [
        rag_search,
        compare_documents,
        generate_bibliography,
        web_search,
        calculator,
        arxiv_search,
        generate_literature_review,
        parallel_document_analysis,
        export_bibliography,
        export_literature_review,       
    ]

def tools_node(state: AgentState):
    """
    The agent's hands - executes the tools the agent requested.
    """
    tools = get_tools()
    tool_registry = {tool.name: tool for tool in tools}
    
    last_message = state["messages"][-1]
    tool_messages = []
    
    # Execute each tool the agent requested
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool = tool_registry.get(tool_name)
        
        if tool:
            print(f"\nExecuting {tool_name}")
            args_preview = str(tool_call["args"])[:100]
            print(f"   Args: {args_preview}")
            
            try:
                result = tool.invoke(tool_call["args"])
                
                # Show preview of result
                result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                print(f"    {result_preview}")
                
                tool_messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_name
                ))
            except Exception as e:
                print(f"   Error: {e}")
                tool_messages.append(ToolMessage(
                    content=f"Error executing {tool_name}: {str(e)}",
                    tool_call_id=tool_call["id"],
                    name=tool_name
                ))
        else:
            print(f"\n{tool_name} not found.")
            tool_messages.append(ToolMessage(
                content=f"Error: '{tool_name}' not found",
                tool_call_id=tool_call["id"],
                name=tool_name
            ))
    
    return {"messages": tool_messages}



# Provider registry - order determines priority
PROVIDER_REGISTRY = []
_current_provider_index = 0

def _build_provider_registry():
    """Build list of available providers from environment variables"""
    global PROVIDER_REGISTRY
    PROVIDER_REGISTRY = []
    
    # Priority 1: Groq (Llama 3) - Testing Stability
    if os.getenv("GROQ_API_KEY"):
        PROVIDER_REGISTRY.append({
            "name": "groq",
            "init": lambda: ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.0,
                max_retries=0 # Fail-fast for fallback
            )
        })

    # Priority 2: Gemini (Flash) - Fallback
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        PROVIDER_REGISTRY.append({
            "name": "gemini",
            "init": lambda: ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                temperature=0.0,
                max_retries=0 # Fail-fast for fallback
            )
        })
    
    # Future providers can be added here (OpenAI, Anthropic, etc.)
    
    if not PROVIDER_REGISTRY:
        raise ValueError(
            "No valid API key found. "
            "Please set GROQ_API_KEY or GEMINI_API_KEY in your .env file"
        )
    
    print(f" Provider Registry: {[p['name'] for p in PROVIDER_REGISTRY]}")

def _get_current_provider():
    """Get the current provider's LLM instance"""
    global _current_provider_index
    if not PROVIDER_REGISTRY:
        _build_provider_registry()
    return PROVIDER_REGISTRY[_current_provider_index]

def _switch_to_next_provider():
    """Cycle to next available provider"""
    global _current_provider_index
    _current_provider_index = (_current_provider_index + 1) % len(PROVIDER_REGISTRY)
    provider = PROVIDER_REGISTRY[_current_provider_index]
    print(f" Switching to: {provider['name']}")
    return provider

def recover_tool_calls(message: AIMessage) -> AIMessage:
    """
    Emergency parser that 'hunts' for JSON tool calls in the message content.
    Essential for Llama-3 models on Groq that often bypass the native tool-calling API.
    """
    if message.tool_calls or not message.content:
        return message
    
    import json, re
    content = message.content.strip()
   
    if content.startswith("{") and content.endswith("}"):
        try:
            data = json.loads(content)
            if "name" in data:
                message.tool_calls = [{
                    "name": data["name"],
                    "args": data.get("parameters", data.get("arguments", data.get("args", {}))),
                    "id": f"recov_{int(time.time())}",
                    "type": "tool_call"
                }]
                print(f"🔧 RECOVERED TOOL CALL (Direct): {data['name']}")
                return message
        except:
            pass

    # Using regex to find all JSON-like blocks
    json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    
    recovered_calls = []
    for block in json_blocks:
        try:
            data = json.loads(block)
            if "name" in data:
                recovered_calls.append({
                    "name": data["name"],
                    "args": data.get("parameters", data.get("arguments", data.get("args", {}))),
                    "id": f"recov_{int(time.time())}_{len(recovered_calls)}",
                    "type": "tool_call"
                })
        except:
            continue
            
    if recovered_calls:
        message.tool_calls = recovered_calls
        print(f"🔧 RECOVERED {len(recovered_calls)} TOOL CALLS from Markdown")
        
    return message

def invoke_with_fallback(messages, tools=None, selected_model=None):
    """
    Invoke LLM with automatic provider switching on rate limits.
    If selected_model is provided, use only that provider.
    """
    global _current_provider_index
    
    if not PROVIDER_REGISTRY:
        _build_provider_registry()
    
    # If a specific model is selected, find its index
    if selected_model:
        selected_model = selected_model.lower()
        for idx, p in enumerate(PROVIDER_REGISTRY):
            if p["name"].lower() == selected_model:
                _current_provider_index = idx
                break
    
    # Only try the current selected provider (Manual override)
    provider = PROVIDER_REGISTRY[_current_provider_index]
    try:
        llm = provider["init"]()
        if tools:
            llm = llm.bind_tools(tools)
        
        response = llm.invoke(messages)
        
        # Use recovery logic for models that output JSON in content
        if tools and isinstance(response, AIMessage):
            response = recover_tool_calls(response)
            
        return response
        
    except Exception as e:
        # Defensive Check: Ensure provider is a dict before logging
        p_name = provider.get("name", "Unknown") if isinstance(provider, dict) else "Index-Error"
        print(f" {p_name} error: {str(e)}")
        # NO AUTOMATIC FALLBACK - Let user switch manually in the UI
        raise e

def _initialize_llm():
    """Initialize the LLM based on current provider (legacy compatibility)"""
    provider = _get_current_provider()
    return provider["init"]()


def llm_node(state: AgentState):
    """The agent's brain"""
    tools = get_tools()

    messages = state["messages"]
    loop_count = state.get("loop_count", 0)

    print(f"LLM Node - Loop: {loop_count}")
    print(f"Messages in state: {len(messages)}")

    # To prevent infinite loops
    if loop_count >= 3:
        return {
            "messages": [AIMessage(content="Proceeding to final answer based on available information.")],
            "loop_count": loop_count
        }
    
    messages = manage_context_window(messages)
    
    # System message - SELECTOR PERSONA (Refined for deterministic tool use)
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        system_msg = SystemMessage(content="""You are Aloysia, an AI Document Reviewer. 

CRITICAL INSTRUCTIONS:
1. Always start by using 'rag_search' to check provided documents.
2. If real-time info is needed, use 'web_search'.
3. If academic papers are needed, use 'arxiv_search'.
4. DO NOT explain yourself. Respond ONLY with a tool call for every user's query
5. If you have enough info, then provide a final answer.

FORMAT REQUIREMENT:
When using a tool, output the raw JSON tool call inside a markdown block:
```json
{"name": "tool_name", "args": {"arg1": "value"}}
```""")
        messages = [system_msg] + messages
    
    try:
        response = invoke_with_fallback(messages, tools=tools, selected_model=state.get("selected_model"))
        
        # LOG RAW CONTENT: See why tools are being skipped
        if hasattr(response, 'content') and response.content:
            print(f"--- LLM Raw Content: {response.content[:200]}... ---")
            
        return {
            "messages": [response],
            "loop_count": loop_count + 1
        }
    except Exception as e:
        error_msg = str(e)
        print(f"LLM Error: {error_msg[:200]}")

        if "tool_use_failed" in error_msg or "Failed to call a function" in error_msg:
            print("Tool call failed. Moving to synthesis with disclaimer.")

            disclaimer = AIMessage(content="""I encountered an issue accessing the document search tools. I an provide general information, but cannot site specific sources at this moment. Kindly rephrase your question or try again.""")

            return {
                "messages": [disclaimer],
                "loop_count": loop_count + 1
            }
        raise



def quality_control_agent_node(state: AgentState):
    """
    Quality Control with web search suggestion capability
    """

    # Getting the last tool results
    tool_messages = [msg for msg in state["messages"]
                     if isinstance(msg, ToolMessage)]
    
    loop_count = state.get("loop_count", 0)


    if not tool_messages or loop_count >= 1:
        # If no tools were used, let's skip QC then
        return {"quality_passed": True, "messages": []}
    
    user_messages = [msg for msg in state["messages"]
                     if isinstance(msg, HumanMessage)]
    original_query = user_messages[0].content if user_messages else ""
    

    recent_results = "\n\n".join([
        f"Tool: {msg.name}\nResult: {msg.content[:800]}..."
        for msg in tool_messages[-2:]
    ])

    # Checking if rag_search returned 'No relevant info'
    no_results = any("No relevant information found" in msg.content
                     for msg in tool_messages if msg.name == "rag_search")

    if no_results:

        feedback = SystemMessage(content=f"""
        The document search found no relevant information for: "{original_query}"

        You should now:
        1. Inform the user that the informatio wasn't found in their documents
        2. Ask if they would like to search the web for this information
        3. Wait for user confirmation before using web_search

        Example response:
        "I couldn't find information about {original_query} in your documents. Would you like me to search the web on this topic?"
        """)

        return {"quality_passed": False, "messages": [feedback]}

    # To save API calls, if the tool returned results and didn't clearly fail, 
    # trust it and skip the LLM evaluation which costs API calls
    
    failure_keywords = ["No relevant information found", "Error searching", "No papers found", "No results found", "Error executing"]
    
    # Check if ANY tool output contains failure keywords
    has_failure = any(keyword in msg.content for msg in tool_messages for keyword in failure_keywords)
    
    if tool_messages and not has_failure:
        print("QC Heuristic: Results look valid. Skipping LLM check to save quota.")
        return {"quality_passed": True, "messages": []}

    eval_prompt = f"""Evaluate search results for: "{original_query}"

Results Retrieved: {recent_results}

Evaluate if these results can answer the question:
- Do the results contain relevant information about "{original_query}"?
- Even if truncated, do they provide useful context on the topic?
- Are there citations or sources included?

Rate 1-10:
- 1-3 = Completely irrelevant, suggest different search terms
- 4-6 = Somewhat relevant, has some useful info
- 7-10 = Highly relevant, answers the question.

Be LENIENT: If results mention the topic or related concepts, score >= 5
Respond with ONLY a number 1-10."""
    
    try:
        response = invoke_with_fallback(eval_prompt)
        content = response.content.strip()
        numbers = re.findall(r'\b([1-9]|10\b)', content)

        if numbers:
            score = int(numbers[0])
            print(f"QC Score: {score}/10")

            quality_passed = score >= 3

            if not quality_passed:
                feedback = SystemMessage(content=f"""
Search scored {score}/10. Try:
1. More specific keywords
2. Different phrasing
3. Check document names
4. If still no results, suggest web search to the user
""")
                
                return {"quality_passed": False, "messages": [feedback]}
            
            print(f"QC Passed: Score {score}/10")
            return {"quality_passed": True, "messages": []}
        else:
            print(f"Could not parse QC score, defaulting to PASS")
            return {"quality_passed": True, "messages": []}

    except Exception as e:
        print(f"QC Error: {e}")
        return {"quality_passed": True, "messages": []}
    
    
def synthesize_final_answer(state: AgentState):
    """
    Second Stage: Synthesize final answer with quality checks
    """
    messages = state["messages"]

    messages = manage_context_window(messages)

    synthesis_prompt = SystemMessage(content="""
Synthesize a final answer based on the available information.


REQUIREMENTS:
- Maximum 300 words
- Include citations if available (Source, Page numbers), 
- Be direct and clear
- Directly answer the question
- If no information found, say so.                                     

Format: According to [Source: filename.pdf, Page: X], ...
""")
    
    messages_with_synthesis = messages + [synthesis_prompt]
    print("Invoking LLM for synthesis...")

    try:
        response = invoke_with_fallback(messages_with_synthesis, selected_model=state.get("selected_model"))

        content = response.content
        print(f"Synthesis Complete. Length: {len(content)} chars")

        # Check if response is short but NOT a question (valid questions come from QC)
        if (not content or len(content.strip()) < 10) and "?" not in content:
            print("Synthesis produced very short response!")
            fallback = AIMessage(content="I apologize, but I couldn't generate a complete response. Please try rephrasing your question.")
            return {"messages": [fallback]}
        
        # Check length
        if len(content) > 5000:
            print("Response too long. Regenerating...")
            feedback = SystemMessage(content="Response too long. Provide a concise 200-word summary.")
            messages_with_feedback = messages_with_synthesis + [response, feedback]
            response = invoke_with_fallback(messages_with_feedback, selected_model=state.get("selected_model"))

        return {"messages": [response]}
    
    except Exception as e:
        print(f"Synthesis Error: {e}")
        
        # FALLBACK: If synthesis fails, construct a response from the tool outputs directly
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        if tool_messages:
            last_tool_result = tool_messages[-1].content
            fallback_content = f"**I found the following information, but couldn't generate a summary due to high traffic:**\n\n{last_tool_result}"
            return {"messages": [AIMessage(content=fallback_content)]}
            
        fallback = AIMessage(content="I apologize, but I encountered an error and couldn't retrieve the results. Please try again.")
        return {"messages": [fallback]}




def should_continue(state: AgentState) -> Literal["tools", "synthesize"]:
    """
    Decision function - determines if agent should use tools or provide final answer.
    """
    last_message = state["messages"][-1]
    
    # If LLM made tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"\nAgent is selecting {len(last_message.tool_calls)} tools...")
        return "tools"
    
    # Otherwise move to synthesis
    print(f"\nAgent is skipping tools. Last msg type: {type(last_message).__name__}")
    if hasattr(last_message, 'content') and last_message.content:
        print(f"--- Skip Answer: {last_message.content[:100]}... ---")
    return "synthesize"


def should_continue_after_qc(state: AgentState) -> Literal["synthesize", "llm"]:
    """Decision after quality control"""
    quality_passed = state.get("quality_passed", True)
    loop_count = state.get("loop_count", 0)
    
    if loop_count >= 2:
        return "synthesize"

    if quality_passed:
        return "synthesize"
    else:
        print("Returning to LLM for refinement")
        return "llm"
    

# Defining the complete agentic RAG workflow
def create_agentic_rag():
    """Build the complete agentic RAG workflow"""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tools_node)
    graph.add_node("quality_control", quality_control_agent_node)
    graph.add_node("synthesize", synthesize_final_answer)
    
    # Set entry point
    graph.set_entry_point("llm")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "synthesize": "synthesize"
        }
    )
    
    # After using tools, go to quality control
    graph.add_edge("tools", "quality_control")

    graph.add_conditional_edges(
        "quality_control",
        should_continue_after_qc,
        {
            "synthesize": "synthesize",
            "llm": "llm"
        }
    )

    graph.add_edge("synthesize", END)
    
    # Compile the graph
    return graph.compile()



# Main
def main():
    """Main function to run the agentic RAG assistant"""
    print("=" * 50)
    print("Agentic RAG Assistant with LangGraph")
    print("=" * 50)
    print("\nInitializing Aloysia...")
    
    try:
        # Create the agent
        agent = create_agentic_rag()
        print("\nAloysia initialized successfully!")
        print("\nFeatures:")
        print("  • Search internal documents (RAG)")
        print("  • Search the web for current info")
        print("  • Search the arXiv for papers according to your query")
        print("  • Perform calculations")
        print("  • Remember conversations")
        print("\nAnd it will automatically choose the best tool to use")
        print("\nType 'exit', or 'x' to quit.\n")
        
        # Maintain conversation state across turns
        conversation_state = {
            "messages": [],
            "quality_passed": True,
            "loop_count": 0,
            "original_query": ""
        }
        
        # Interactive loop
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit commands BEFORE processing
            if user_input.lower() in ["exit", "x"]:
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user message to conversation
            conversation_state["messages"].append(HumanMessage(content=user_input))
            conversation_state["original_query"] = user_input
            conversation_state["loop_count"] = 0
            
            print("\nProcessing...")
            
            try:
                # Invoke the agent with conversation history
                result = agent.invoke(conversation_state)
                
                # Update conversation with all new messages (tool calls, results, final answer)
                conversation_state["messages"] = result["messages"]
                
                # Get the final response (last message)
                final_message = result["messages"][-1]
                
                print("-" * 60)
                print(f"\nAssistant: {final_message.content}\n")
                
            except Exception as e:
                error_msg = str(e)
                print(f"\nError during processing: {error_msg}")
                
                # Provide helpful error context
                if "tool_use_failed" in error_msg:
                    print("\nTry:")
                    print("  - Rephrasing your question")
                    print("  - Making it more specific or simpler")
                else:
                    print("Continuing conversation...\n")
                
                # Remove the failed user message to prevent cascading errors
                if conversation_state["messages"]:
                    conversation_state["messages"].pop()
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nInitialization Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set GROQ_API_KEY or GEMINI_API_KEY in .env")
        print("  2. (Optional) Set TAVILY_API_KEY for web search")
        print("  3. Documents in the 'data' folder")


if __name__ == "__main__":
    main()
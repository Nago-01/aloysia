import os, traceback
from typing import List, Dict, Any
from supabase.client import create_client, Client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import CrossEncoder

class VectorDB:
    """
    Cloud-ready Vector database using Supabase and Gemini Embeddings.
    """
    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the cloud vector database.
        """
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.table_name = os.getenv("SUPABASE_TABLE_NAME", "documents")
        
        if not self.supabase_url or not self.supabase_key:
            print("WARNING: Supabase credentials missing in .env")

        # Initialize Supabase client
        self.client: Client = create_client(self.supabase_url, self.supabase_key)

        # Offload embeddings to Gemini API (Faster, lightweight)
        print("Initializing Gemini Embeddings API...")
        
        # FIX: Ensure an asyncio event loop exists for Google's async gRPC client
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        # Initialize cross-encoder for re-ranking (Keep local for quality)
        # Note: This is 100x smaller than the embedding model
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize Vector Store
        self.vector_store = SupabaseVectorStore(
            client=self.client,
            embedding=self.embeddings,
            table_name=self.table_name,
            query_name="match_documents" # Expected RPC name in Supabase
        )

        print(f"Cloud Vector DB initialized (Supabase: {self.supabase_url})")



    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Chunk text using Semantic Chunking based on Gemini embeddings."""
        
        text_splitter = SemanticChunker(
            self.embeddings, 
            breakpoint_threshold_type="percentile"
        )
        try:
            chunks = text_splitter.split_text(text)
            return chunks
        except Exception as e:
            print(f"Error in semantic chunking: {e}. Fallback to simple split.")
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    
    def add_doc(self, documents: List, user_id: str = "default_user") -> None:
        """
        Add documents to Supabase with metadata.
        """
        all_chunks = []
        all_metadatas = []

        for doc in documents:
            text = doc.get("content", "")
            metadata = doc.get("metadata", {}).copy()
            
            # Security: Add user_id to metadata for filtering
            metadata["user_id"] = user_id

            if len(text) < 1000:
                chunks = [text]
            else:
                chunks = self.chunk_text(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append(metadata)

        # Store in Supabase
        if all_chunks:
            print(f"Uploading {len(all_chunks)} chunks to Supabase...")
            
            # SANITIZATION: PostgreSQL cannot store \u0000 (null characters)
            sanitized_chunks = [chunk.replace("\u0000", "") for chunk in all_chunks]
            
            self.vector_store.add_texts(
                texts=sanitized_chunks,
                metadatas=all_metadatas
            )
        
        print(f"Synchronization complete: {len(documents)} docs added to cloud.")



    def search(self, query: str, n_results: int = 5, use_reranking: bool = True, user_id: str = "default_user") -> Dict[str, Any]:
        """
        Search cloud DB with direct RPC call to bypass buggy SDK.
        """
        try:
            # 1. Generate embedding for the query (with retry logic)
            query_embedding = None
            for attempt in range(3):
                try:
                    query_embedding = self.embeddings.embed_query(query)
                    break
                except Exception as emb_e:
                    if "429" in str(emb_e) and attempt < 2:
                        print(f"⚠️ Embedding Rate Limit hit. Retrying in {attempt + 1}s...")
                        import time
                        time.sleep(attempt + 1)
                        continue
                    else:
                        print(f"⚠️ Embedding API Error: {emb_e}. Skipping RAG search.")
                        return {"documents": [], "metadatas": [], "distances": [], "ids": [], "citations": []}

            # 2. Call the 'match_documents' RPC function directly
            print(f"DEBUG: Searching Supabase for '{query[:30]}...' (user_id: {user_id})")
            
            response = self.client.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.1, # Slightly higher than 0.0 for basic signal
                    "match_count": n_results * 4 if use_reranking else n_results, # Over-sample for local reranker
                    "filter": {"user_id": user_id}
                }
            ).execute()

            if not response.data or len(response.data) == 0:
                print(f"DEBUG: Supabase query returned 0 rows for user_id={user_id}")
                return {"documents": [], "metadatas": [], "distances": [], "ids": [], "citations": []}
            
            print(f"DEBUG: Successfully retrieved {len(response.data)} chunks from cloud.")
            
            # Unpack results
            documents = [row["content"] for row in response.data]
            metadatas = [row["metadata"] for row in response.data]
            scores = [row["similarity"] for row in response.data]

            # Re-rank using local cross-encoder for precision
            if use_reranking and len(documents) > 0:
                pairs = [[query, doc] for doc in documents]
                rerank_scores = self.reranker.predict(pairs)

                sorted_indices = sorted(
                    range(len(rerank_scores)),
                    key=lambda i: rerank_scores[i],
                    reverse=True
                )[:n_results]

                documents = [documents[i] for i in sorted_indices]
                metadatas = [metadatas[i] for i in sorted_indices]
                scores = [rerank_scores[i] for i in sorted_indices]

            return {
                "documents": documents,
                "metadatas": metadatas,
                "distances": scores,
                "citations": [self._format_citation(m) for m in metadatas]
            }
        
        except Exception as e:
            print(f"Error during cloud search: {e}")
            traceback.print_exc()
            return {"documents": [], "metadatas": [], "distances": [], "ids": [], "citations": []}


    def list_all_metadata(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """
        Retrieve all unique document metadata for this user.
        Used for generating bibliography.
        """
        try:
            # Note: For large DBs, we'd want to use distinct on metadata->>source
            # But for Beta, we just grab all and filter in Python
            response = self.client.from_(self.table_name).select("metadata").eq("metadata->>user_id", user_id).execute()
            
            if not response.data:
                return []
            
            # Return list of metadata dicts
            return [row["metadata"] for row in response.data]
            
        except Exception as e:
            print(f"Error listing metadata: {e}")
            return []
        

    def _format_citation(self, metadata: Dict[str, Any]) -> str:
        source = metadata.get("source", "Unknown Source")
        page = metadata.get("page_number", "N/A")
        author = metadata.get("author", "Unknown Author")
        title = metadata.get("title", "Untitled")
        section = metadata.get("section", "")

        citation_parts = [f"Source: {source}"]
        if page != "N/A": citation_parts.append(f"Page: {page}")
        if author and author != "Unknown Author": citation_parts.append(f"Author: {author}")
        if title and title != source: citation_parts.append(f"Title: {title}")
        if section: citation_parts.append(f"Section: {section}")

        return ", ".join(citation_parts)
    

    def add_arxiv_papers(self, papers: list, user_id: str = "default_user") -> int:
        documents = []
        for paper in papers:
            content = f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nAbstract: {paper['abstract']}"
            metadata = {
                "source": paper['url'],
                "title": paper['title'],
                "author": ', '.join(paper['authors'][:3]),
                "type": "arxiv_paper",
                "user_id": user_id
            }
            documents.append({"content": content, "metadata": metadata})
        
        if documents:
            self.add_doc(documents, user_id=user_id)
        return len(documents)
import os, traceback
from typing import List, Dict, Any
from supabase.client import create_client, Client
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_experimental.text_splitter import SemanticChunker
class VectorDB:
    """
    Cloud-ready Vector database using Supabase and Ultra-Light FastEmbed.
    """
    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the cloud vector database.
        """
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_KEY"))
        self.table_name = os.getenv("SUPABASE_TABLE_NAME", "documents")
        
        # Debugging for Render logs
        if not self.supabase_url:
            print("❌ ERROR: SUPABASE_URL is missing from environment variables!")
        if not self.supabase_key:
            print("❌ ERROR: SUPABASE_KEY/SERVICE_ROLE_KEY is missing!")
            
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Required Supabase environment variables (SUPABASE_URL, SUPABASE_KEY) are missing.")

        # Initialize Supabase client
        self.client: Client = create_client(self.supabase_url, self.supabase_key)

        # Use FastEmbed (Zero-Torching, Ultra-Lightweight for Render Free Tier)
        print("Initializing FastEmbed...")
        self.embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="/tmp/fastembed_cache"
        )

        if os.getenv("DISABLE_RERANKER", "false").lower() == "true":
            print("Reranker is DISABLED via Env Var")
            self.reranker = None
        else:
            try:
                print("Checking for Cross-Encoder libraries...")
                from sentence_transformers import CrossEncoder
                print("Initializing local Cross-Encoder...")
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                print(f"Warning: Cross-Encoder skipped or failed ({e}). Falling back to standard cosine similarity.")
                self.reranker = None
        
        # Initialize Vector Store
        self.vector_store = SupabaseVectorStore(
            client=self.client,
            embedding=self.embeddings,
            table_name=self.table_name,
            query_name="match_documents" # Expected RPC name in Supabase
        )
        print(f"Supabase Initialized: {self.supabase_url}")



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
            if isinstance(doc, dict):
                text = doc.get("content", "")
                metadata = doc.get("metadata", {}).copy()
            else:
                # Handle LangChain Document objects
                text = getattr(doc, "page_content", "")
                metadata = getattr(doc, "metadata", {}).copy()
            
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
                        print(f"Embedding Rate Limit hit. Retrying in {attempt + 1}s...")
                        import time
                        time.sleep(attempt + 1)
                        continue
                    else:
                        print(f"Embedding API Error: {emb_e}. Skipping RAG search.")
                        return {"documents": [], "metadatas": [], "distances": [], "ids": [], "citations": []}

            # 2. Call the 'match_documents' RPC function directly
            print(f"DEBUG: Searching Supabase for '{query[:30]}...' (user_id: {user_id})")
            
            # DIAGNOSTIC: Check if user has ANY documents first
            try:
                count_resp = self.client.from_(self.table_name).select("id", count="exact").eq("metadata->>user_id", user_id).limit(1).execute()
                total_chunks = count_resp.count if hasattr(count_resp, 'count') else 0
                print(f"📊 [LIBRARY_STAT] User {user_id} has {total_chunks} total chunks indexed.")
            except Exception:
                pass

            print(f"DEBUG: Executing RPC match_documents with filter: {{'user_id': '{user_id}'}} (Type: {type(user_id)})")
            
            response = self.client.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.1, 
                    "match_count": n_results * 4 if use_reranking else n_results,
                    "filter": {"user_id": user_id}
                }
            ).execute()

            if hasattr(response, 'error') and response.error:
                print(f"❌ Supabase RPC Error: {response.error}")

            if not response.data:
                print(f"🔍 [SEARCH] 0 chunks found for user_id: {user_id}")
                return {"documents": [], "metadatas": [], "distances": [], "ids": [], "citations": []}
            
            print(f"✅ [SEARCH] {len(response.data)} chunks found for user_id: {user_id}")
            
            # Unpack results
            documents = [row["content"] for row in response.data]
            metadatas = [row["metadata"] for row in response.data]
            scores = [row["similarity"] for row in response.data]

            # FILTER: Skip bibliography/references chunks (low-value for Q&A)
            # Patterns that indicate reference/bibliography sections
            REFERENCE_PATTERNS = [
                r'^\d+\.\s+[A-Z][a-z]+\s+[A-Z]{1,2},',  # "1. Smith J, ..."
                r'^\[\d+\]',  # "[1] Reference..."
                r'^References\s*$',
                r'^Bibliography\s*$',
                r'et\s+al\.\s*\(\d{4}\)',  # "et al. (2020)"
                r'doi:\s*10\.',  # DOI patterns
                r'PMID:\s*\d+',  # PubMed IDs
            ]
            import re
            
            def is_reference_chunk(content: str) -> bool:
                """Check if chunk appears to be from references/bibliography section."""
                # Check first 200 chars for reference patterns
                sample = content[:200]
                for pattern in REFERENCE_PATTERNS:
                    if re.search(pattern, sample, re.IGNORECASE | re.MULTILINE):
                        return True
                # Also check if most lines look like citations (short with years)
                lines = content.split('\n')[:5]
                citation_like = sum(1 for l in lines if re.search(r'\(\d{4}\)', l) or re.search(r'\d{4};\d+', l))
                return citation_like >= 3
            
            # Filter out reference chunks
            filtered_data = [
                (d, m, s) for d, m, s in zip(documents, metadatas, scores)
                if not is_reference_chunk(d)
            ]
            
            if filtered_data:
                documents, metadatas, scores = zip(*filtered_data)
                documents, metadatas, scores = list(documents), list(metadatas), list(scores)
                print(f"DEBUG: After filtering references: {len(documents)} chunks remain.")
            else:
                print("DEBUG: All chunks were references. Using original results.")

            # Re-rank using local cross-encoder for precision
            if use_reranking and self.reranker and len(documents) > 0:

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
            else:
                # FALLBACK: CrossEncoder unavailable - use similarity threshold filtering
                # Filter out low-quality results (cosine sim < 0.15) and take top n_results
                MIN_SIMILARITY = 0.15
                filtered = [(d, m, s) for d, m, s in zip(documents, metadatas, scores) if s >= MIN_SIMILARITY]
                
                if filtered:
                    documents, metadatas, scores = zip(*filtered[:n_results])
                    documents, metadatas, scores = list(documents), list(metadatas), list(scores)
                else:
                    # If all results are low quality, still return top n_results but warn
                    documents = documents[:n_results]
                    metadatas = metadatas[:n_results]
                    scores = scores[:n_results]
                    print(f"Warning: All {len(documents)} results below similarity threshold ({MIN_SIMILARITY}).")

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

    def link_user(self, telegram_id: str, email: str) -> bool:
        """Associate a Telegram ID with an email in Supabase."""
        try:
            data = {
                "telegram_id": str(telegram_id),
                "email": email.lower().strip()
            }
            # Upsert mapping
            self.client.from_("user_mappings").upsert(data).execute()
            return True
        except Exception as e:
            print(f"Error linking user: {e}")
            return False

    def get_mapped_user(self, telegram_id: str) -> str:
        """Get the email associated with a Telegram ID. Returns telegram_id if not found."""
        try:
            response = self.client.from_("user_mappings").select("email").eq("telegram_id", str(telegram_id)).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]["email"]
        except Exception as e:
            # Table might not exist yet
            pass
        
        return str(telegram_id)
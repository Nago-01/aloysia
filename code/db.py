import os, chromadb, traceback
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

class VectorDB:
    """
    Vector database with enhanced citation tracking.
    """
    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "aloysia_knowledge"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Initialize cross-encoder for re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Creating collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")



    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Chunk text using Semantic Chunking based on embedding similarity."""
        
        # Wrap our existing model
        embeddings_wrapper = LocalHuggingFaceEmbeddings(self.embedding_model)
        
        # Create semantic chunker
        # 'percentile' threshold helps determine where meaning shifts occur
        text_splitter = SemanticChunker(
            embeddings_wrapper, 
            breakpoint_threshold_type="percentile"
        )
        try:
            chunks = text_splitter.split_text(text)
            return chunks
        except Exception as e:
            print(f"Error in semantic chunking: {e}. Fallback to simple split.")
            # Simple fallback if semantic fails (e.g. text too short)
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    
    def add_doc(self, documents: List) -> None:
        """
        Add documents with enhanced metadata tracking.
        """

        for i, doc in enumerate(documents):
            text = doc.get("content", "")
            metadata = doc.get("metadata", {})


            if len(text) < 1000:
                chunks = [text]
            else:
                chunks = self.chunk_text(text, chunk_size=800)

            # Unique IDs and enriched metadata for each chunk
            chunk_ids = []
            chunk_metadatas = []

            for j, chunk in enumerate(chunks):
                chunk_id = f"doc_{i}_page_{metadata.get('page_number', 0)}_chunk_{j}"
                chunk_ids.append(chunk_id)

                # Add chunk-specific metadata
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = j
                chunk_metadata["total_chunks"] = len(chunks)
                chunk_metadata["chunk_id"] = chunk_id

                chunk_metadatas.append(chunk_metadata)



            # Generate embeddings for all chunks
            embeddings = self.embedding_model.encode(chunks)

            # Store in ChromaDB
            self.collection.add(
                documents=chunks,
                metadatas=chunk_metadatas,
                embeddings = embeddings.tolist() if not isinstance(embeddings, list) else embeddings,
                ids=chunk_ids
            )
        
        print(f"{len(documents)} document chunks have been added to the vector database.")



    def search(self, query: str, n_results: int = 5, use_reranking: bool = True) -> Dict[str, Any]:
        """
        Search with enhanced citation information and optional cross-encoder re-ranking.

        Args: 
            query: Search query
            n_results: Number of results to return
            use_reranking: Whether to apply cross-encoder

        Returns results with full metadata for citation tracking. 
        """

        try:
            # Encode the query to get its embedding
            query_embedding = self.embedding_model.encode([query])

            # Retrieve more results for reranking if needed
            initial_results = n_results * 3 if use_reranking else n_results

            # Perform similarity search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=initial_results,
            )

            # Handling empty results
            if not results:
                return {"documents": [], "metadatas": [], "distances": [], "ids": [], "citations": []}
            
            # Unpack results
            documents = results.get("documents", [])[0]
            metadatas = results.get("metadatas", [])[0]
            distances = results.get("distances", [])[0]
            ids = results.get("ids", [])[0]

            # Re-rank using cross-encoder if enabled
            if use_reranking and len(documents) > 0:
                # Create query-document pairs
                pairs = [[query, doc] for doc in documents]

                # Get cross-encoder scores
                rerank_scores = self.reranker.predict(pairs)

                # Sorting by rerank scores
                sorted_indices = sorted(
                    range(len(rerank_scores)),
                    key=lambda i: rerank_scores[i],
                    reverse=True
                )[:n_results]

                # Reorder results based on reranking
                documents = [documents[i] for i in sorted_indices]
                metadatas = [metadatas[i] for i in sorted_indices]
                distances = [rerank_scores[i] for i in sorted_indices]
                ids = [ids[i] for i in sorted_indices]

            # Create citations string for each result
            citations = [self._format_citation(metadata) for metadata in metadatas]

            # Return structured dictionary
            return {
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances,
                "ids": ids,
                "citations": citations
            }
        
        except Exception as e:
            print(f"Error during search: {e}")
            traceback.print_exc()
            return {"documents": [], "metadatas": [], "distances": [], "ids": [], "citations": []}
        

    def _format_citation(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata into a citation string.
        
        Example: "Source: document.pdf, Page: 5, Author: John Doe, Section: Introduction
        """
        source = metadata.get("source", "Unknown Source")
        page = metadata.get("page_number", "N/A")
        author = metadata.get("author", "Unknown Author")
        title = metadata.get("title", "Untitled")
        section = metadata.get("section", "")

        citation_parts = [f"Source: {source}"]

        if page != "N/A":
            citation_parts.append(f"Page: {page}")

        if author and author != "Unknown Author":
            citation_parts.append(f"Author: {author}")

        if title and title != source:
            citation_parts.append(f"Title: {title}")

        if section:
            citation_parts.append(f"Section: {section}")

        return ", ".join(citation_parts)
        

    def add_arxiv_papers(self, papers: list) -> int:
        """
        Add arXiv papers to the vector database.
        
        Args:
            papers: List of paper dictionaries with keys:
                - title, authors, abstract, url, pdf_url, published, categories
        
        Returns:
            Number of papers added
        """
        documents = []
        
        for paper in papers:
            # Create searchable content from paper metadata
            content = f"""Title: {paper['title']}

Authors: {', '.join(paper['authors'])}

Abstract: {paper['abstract']}

Categories: {', '.join(paper.get('categories', []))}
"""
            
            metadata = {
                "source": paper['url'],
                "title": paper['title'],
                "author": ', '.join(paper['authors'][:3]),
                "type": "arxiv_paper",
                "page_number": 1,
                "published_date": paper['published'],
                "pdf_url": paper['pdf_url'],
                "categories": ', '.join(paper.get('categories', []))
            }
            
            documents.append({
                "content": content,
                "metadata": metadata
            })
        
        if documents:
            self.add_doc(documents)
        
        return len(documents)
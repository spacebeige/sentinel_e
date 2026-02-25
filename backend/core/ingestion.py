import os
import logging

logger = logging.getLogger("IngestionEngine")

# ── Feature flag: disable heavy ML deps in cloud deployment ──
USE_LOCAL_INGESTION = os.getenv("USE_LOCAL_INGESTION", "false").lower() == "true"


class IngestionEngine:
    """
    RAG ingestion engine with lazy-loaded ML dependencies.
    Heavy deps (sentence-transformers, FAISS, torch) are only loaded
    when USE_LOCAL_INGESTION=true AND a method that needs them is called.
    """

    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.doc_store = []
        self.index_path = "data/faiss_index"
        self._initialized = False

        if USE_LOCAL_INGESTION:
            self._lazy_init()
        else:
            logger.info("IngestionEngine running in cloud mode (no local embeddings/FAISS)")

    def _lazy_init(self):
        """Load heavy ML dependencies only when explicitly needed."""
        if self._initialized:
            return
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self._load_index()
            self._initialized = True
            logger.info("IngestionEngine ML dependencies loaded successfully")
        except ImportError as e:
            logger.warning(f"ML dependencies not available: {e}")
        except Exception as e:
            logger.warning(f"IngestionEngine initialization failed: {e}")

    def _load_index(self):
        """Load existing FAISS index if available."""
        if os.path.exists(self.index_path) and self.embeddings:
            try:
                from langchain_community.vectorstores import FAISS
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")

    def save_index(self):
        """Save FAISS index to disk."""
        if self.vector_store:
            self.vector_store.save_local(self.index_path)

    def process_pdf(self, file_path):
        """Parse PDF, chunk text, and build FAISS index."""
        if not self._initialized:
            self._lazy_init()
        if not self._initialized:
            raise RuntimeError("IngestionEngine ML dependencies not available. Set USE_LOCAL_INGESTION=true and install deps.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        from langchain_community.document_loaders import PyMuPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS

        # 1. Load PDF
        loader = PyMuPDFLoader(file_path)
        
        # 2. Split Text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        try:
            pages = loader.load_and_split(splitter)
        except Exception as e:
             raise RuntimeError(f"Error loading PDF: {str(e)}")

        # Clean small chunks
        clean_docs = [d for d in pages if len(d.page_content.strip()) > 100]
        
        if not clean_docs:
             return 0

        self.doc_store.extend(clean_docs)

        # 3. Build/Update Vector Store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(clean_docs, self.embeddings)
        else:
            self.vector_store.add_documents(clean_docs)
            
        self.save_index()
        return len(clean_docs)

    def retrieve_context(self, query, k=3):
        if not self.vector_store:
            return []
        
        # Retrieve top k documents
        docs = self.vector_store.similarity_search(query, k=k)
        return docs
    
    def add_text_memory(self, text: str, metadata: dict = None):
        """Add text directly to memory (for learnings)."""
        if not text: return
        if not self._initialized:
            self._lazy_init()
        if not self.embeddings:
            logger.warning("Cannot add text memory: embeddings not available")
            return

        try:
            from langchain_core.documents import Document
        except ImportError:
            from langchain.docstore.document import Document

        from langchain_community.vectorstores import FAISS
            
        doc = Document(page_content=text, metadata=metadata or {})
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vector_store.add_documents([doc])
        self.save_index()

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class IngestionEngine:
    def __init__(self):
        # Initialize embeddings on CPU
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.doc_store = []
        self.index_path = "data/faiss_index"
        self._load_index()

    def _load_index(self):
        """Load existing FAISS index if available."""
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
                print(f"Loaded FAISS index from {self.index_path}")
            except Exception as e:
                print(f"Failed to load FAISS index: {e}")

    def save_index(self):
        """Save FAISS index to disk."""
        if self.vector_store:
            self.vector_store.save_local(self.index_path)

    def process_pdf(self, file_path):
        """Parse PDF, chunk text, and build FAISS index."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

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
        try:
            from langchain_core.documents import Document
        except ImportError:
            from langchain.docstore.document import Document
            
        doc = Document(page_content=text, metadata=metadata or {})
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vector_store.add_documents([doc])
        self.save_index()

# import torch
# import torch.nn.functional as F
# from transformers import AutoModel
# from typing import List, Tuple
# import numpy as np

# class EmbeddingRetriever:
#     def __init__(self, model_name: str = 'nvidia/NV-Embed-v2', device: str = None):
#         """
#         Initialize the embedding-based retriever
        
#         Args:
#             model_name: Name of the pretrained model to use
#             device: Device to run the model on ('cuda' or 'cpu')
#         """
#         self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
#         self.embed_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
#         self.embed_model.to(self.device)
#         self.embed_model.eval()
        
#         self.documents = []
#         self.embeddings = None
        
#     def _get_embedding(self, text: str) -> torch.Tensor:
#         """
#         Get embedding for a single text
        
#         Args:
#             text: Input text string
            
#         Returns:
#             Normalized embedding tensor
#         """
#         with torch.no_grad():
#             embedding = self.embed_model.encode(
#                 [text], 
#                 instruction="", 
#                 max_length=32768
#             )
#             # Normalize the embedding
#             embedding = F.normalize(embedding, p=2, dim=1)
#         return embedding
                    
#     def fit(self, documents: List[str], batch_size: int = 2):
#         """
#         Process and store embeddings for a collection of documents
        
#         Args:
#             documents: List of document strings
#             batch_size: Size of batches for processing
#         """
#         self.documents = documents
#         self.embeddings = []
#         for doc in documents:
#             embedding = self._get_embedding(doc)
#             self.embeddings.append(embedding)
        
#         self.embeddings = torch.cat(self.embeddings, dim=0)
    
#     def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[int, str, float]]:
#         """
#         Retrieve top-k most similar documents for a query
        
#         Args:
#             query: Query string
#             top_k: Number of documents to retrieve (default: 1)
            
#         Returns:
#             List of tuples (doc_id, document, similarity_score) sorted by similarity
#         """
#         # Get query embedding
#         query_embedding = self._get_embedding(query)
        
#         # Compute similarities with all documents
#         similarities = F.cosine_similarity(
#             query_embedding.unsqueeze(0),  # [1, embedding_dim]
#             self.embeddings  # [n_docs, embedding_dim]
#         )
        
#         # Get top-k similar documents
#         top_k = min(top_k, len(self.documents))
#         top_scores, top_indices = torch.topk(similarities, k=top_k)
        
#         # Prepare results
#         results = [
#             (idx.item(), self.documents[idx.item()], score.item())
#             for idx, score in zip(top_indices, top_scores)
#         ]
        
#         return results


# if __name__ == "__main__":
#     # Example usage
#     documents = [
#         "This is the first document.",
#         "This document is the second document.",
#         "And this is the third one.",
#         "Is this the first document?",
#     ]
    
#     # Initialize and fit the embedding retriever
#     retriever = EmbeddingRetriever()
#     retriever.fit(documents)
    
#     # Test single query retrieval
#     query = "find the first document"
#     results = retriever.retrieve(query, top_k=1)
    
#     print("Single Query Results:")
#     for i, doc, score in results:
#         print(f"Document {i}: {doc} (Similarity score: {score:.4f})")

import torch
import torch.nn.functional as F
from transformers import AutoModel
from typing import List, Tuple
import numpy as np

class EmbeddingRetriever:
    def __init__(self, model_name: str = 'nvidia/NV-Embed-v2', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.embed_model.to(self.device)
        self.embed_model.eval()
        
        self.documents = []
        self.embeddings = None
        
    def _get_embedding(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.embed_model.encode(
                [text], 
                instruction="", 
                max_length=32768
            )
            # Normalize the embedding
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
                    
    def fit(self, documents: List[str]):
        """Process and store embeddings for documents"""
        self.documents = documents
        embeddings_list = []
        
        for doc in documents:
            embedding = self._get_embedding(doc)
            embeddings_list.append(embedding)
        
        # Stack all embeddings into a single tensor [n_docs, embedding_dim]
        self.embeddings = torch.cat(embeddings_list, dim=0)
    
    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[int, str, float]]:
        """Retrieve top-k most similar documents"""
        # Get query embedding [1, embedding_dim]
        query_embedding = self._get_embedding(query)
        
        # Compute dot product between query and all documents
        # This is equivalent to cosine similarity since vectors are normalized
        similarities = torch.mm(query_embedding, self.embeddings.t()).squeeze(0)
        
        # Get top-k similar documents
        top_k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(similarities, k=top_k)
        
        # Prepare results
        results = [
            (idx.item(), self.documents[idx.item()], score.item())
            for idx, score in zip(top_indices, top_scores)
        ]
        
        return results


if __name__ == "__main__":
    documents = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
        "This is the last document."
    ]
    
    print("Initializing retriever...")
    retriever = EmbeddingRetriever()
    
    print("Fitting documents...")
    retriever.fit(documents)
    
    query = "find the first document"
    print(f"\nQuerying: '{query}'")
    
    print("\nComputing similarities...")
    results = retriever.retrieve(query, top_k=1)
    
    print("\nResults:")
    for i, doc, score in results:
        print(f"Document {i}: {doc} (Similarity score: {score:.4f})")
            
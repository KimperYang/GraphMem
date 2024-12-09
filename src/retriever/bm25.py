from collections import Counter
import math
import re
from typing import List, Dict, Tuple

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever with parameters
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}  # Document frequencies
        self.doc_lengths = []  # Document lengths
        self.avg_doc_length = 0
        self.total_docs = 0
        self.documents = []  # Original documents
        self.doc_terms = []  # Tokenized documents
        
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text by converting to lowercase and splitting into terms
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed terms
        """
        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        terms = re.findall(r'\w+', text)
        return terms
    
    def fit(self, documents: List[str]):
        """
        Fit the BM25 model on a collection of documents
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        self.total_docs = len(documents)
        
        # Preprocess all documents
        self.doc_terms = [self.preprocess(doc) for doc in documents]
        
        # Calculate document lengths
        self.doc_lengths = [len(terms) for terms in self.doc_terms]
        self.avg_doc_length = sum(self.doc_lengths) / self.total_docs
        
        # Calculate document frequencies
        self.doc_freqs = {}
        for doc_terms in self.doc_terms:
            term_set = set(doc_terms)
            for term in term_set:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
    
    def get_scores(self, query: str) -> List[float]:
        """
        Calculate BM25 scores for a query against all documents
        
        Args:
            query: Query string
            
        Returns:
            List of BM25 scores for each document
        """
        query_terms = self.preprocess(query)
        scores = [0.0] * self.total_docs
        
        for term in query_terms:
            if term not in self.doc_freqs:
                continue
                
            # Calculate IDF
            idf = math.log((self.total_docs - self.doc_freqs[term] + 0.5) / 
                          (self.doc_freqs[term] + 0.5) + 1.0)
            
            for doc_id, doc_terms in enumerate(self.doc_terms):
                # Calculate term frequency
                term_freq = Counter(doc_terms)[term]
                
                # Calculate document length normalization
                doc_len_norm = 1 - self.b + self.b * (self.doc_lengths[doc_id] / self.avg_doc_length)
                
                # Calculate BM25 score for this term
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * doc_len_norm
                scores[doc_id] += idf * (numerator / denominator)
        
        return scores
    
    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[int, str, float]]:
        """
        Retrieve top-k most relevant documents for a query
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve (default: 1)
            
        Returns:
            List of tuples (doc_id, document, score) sorted by relevance
        """
        scores = self.get_scores(query)
        
        # Sort documents by score
        doc_scores = list(enumerate(zip(self.documents, scores)))
        doc_scores.sort(key=lambda x: x[1][1], reverse=True)
        
        # Return top-k results
        results = [(i, doc, score) for i, (doc, score) in doc_scores[:top_k]]
        return results


if __name__ == "__main__":
    # Example usage
    documents = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    
    retriever = BM25Retriever()
    retriever.fit(documents)
    
    query = "find the first document"
    results = retriever.retrieve(query, top_k=1)
    
    for i, doc, score in results:
        print(f"Document {i}: {doc} (BM25 score: {score})")
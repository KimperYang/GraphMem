import networkx as nx
from sentence_transformers import SentenceTransformer, util

class SemanticKnowledgeGraph:
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.8):
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
    
    def add_node(self, node):
        if node not in self.graph:
            self.graph.add_node(node)
    
    def add_edge(self, node1, node2, relation):
        self.add_node(node1)
        self.add_node(node2)
        
        new_relation_emb = self.model.encode(relation)
        
        if self.graph.has_edge(node1, node2):
            existing_relation = self.graph[node1][node2].get('relation', None)
            existing_relation_emb = self.graph[node1][node2]['relation_embedding']

            sim = util.cos_sim(new_relation_emb, existing_relation_emb).item()
            
            if sim < self.similarity_threshold:
                old_rel = existing_relation
                self.graph[node1][node2]['relation'] = relation
                self.graph[node1][node2]['relation_embedding'] = new_relation_emb
                return {
                    'updated': True,
                    'message': f"Edge ({node1}, {node2}) relation updated from '{old_rel}' to '{relation}'",
                    'similarity': sim
                }
            else:
                return {
                    'updated': False,
                    'message': f"Edge ({node1}, {node2}) has a semantically similar relation",
                    'similarity': sim
                }
        else:
            self.graph.add_edge(node1, node2, relation=relation, relation_embedding=new_relation_emb)
            return {
                'updated': False,
                'message': f"Edge ({node1}, {node2}) with relation '{relation}' added",
                'similarity': 1.0
            }

    def query_edge(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2]['relation']
        else:
            return None

if __name__ == "__main__":   
    kg = SemanticKnowledgeGraph()

    kg.add_edge("A", "B", "works with")
    kg.add_edge("A", "B", "collaborates alongside")
    kg.add_edge("A", "B", "is a competitor of")
    print("Current relation between A and B:", kg.query_edge("A", "B"))

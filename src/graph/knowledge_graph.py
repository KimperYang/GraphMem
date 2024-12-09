import networkx as nx
from sentence_transformers import SentenceTransformer, util

class SemanticKnowledgeGraph:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer(model_name)
    
    def add_node(self, node):
        if node not in self.graph:
            self.graph.add_node(node)
    
    def add_edge(self, node1, node2, relation, overwrite=False):
        self.add_node(node1)
        self.add_node(node2)
        
        new_relation_emb = self.model.encode(relation)
        
        if self.graph.has_edge(node1, node2):
            existing_relation = self.graph[node1][node2].get('relation', None)
            if existing_relation == relation:
                return {
                    'conflict': True,
                    'message': f"Edge ({node1}, {node2}) with relation '{relation}' already exists",
                }
                
            else:
                self.graph.add_edge(node1, node2, relation=relation, relation_embedding=new_relation_emb)
                return {
                    'conflict': False,
                    'message': f"Edge ({node1}, {node2}) with relation '{relation}' added",
                }
        else:
            self.graph.add_edge(node1, node2, relation=relation, relation_embedding=new_relation_emb)
            return {
                'updated': False,
                'message': f"Edge ({node1}, {node2}) with relation '{relation}' added",
            }

    def query(self, node1=None, node2=None, relation=None, top_k=1):
        query_node1_emb = self.model.encode(node1) if node1 else None
        query_node2_emb = self.model.encode(node2) if node2 else None
        query_relation_emb = self.model.encode(relation) if relation else None

        matches = []
        scores = []

        for n1, n2, attrs in self.graph.edges(data=True):
            score = 0

            # Compare node1
            if query_node1_emb is not None:
                score += util.cos_sim(query_node1_emb, self.graph.nodes[n1]['embedding']).item()

            # Compare node2
            if query_node2_emb is not None:
                score += util.cos_sim(query_node2_emb, self.graph.nodes[n2]['embedding']).item()

            # Compare relation
            if query_relation_emb is not None:
                score += util.cos_sim(query_relation_emb, attrs['relation_embedding']).item()

            matches.append((n1, n2))
            scores.append(score)

        top_k_matches = [matches[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]]
        top_k_scores = [scores[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]]
        
        return top_k_matches, top_k_scores

if __name__ == "__main__":   
    kg = SemanticKnowledgeGraph()

    kg.add_edge("A", "B", "works with")
    kg.add_edge("A", "B", "collaborates alongside")
    kg.add_edge("A", "B", "is a competitor of")
    print("Current relation between A and B:", kg.query_edge("A", "B"))

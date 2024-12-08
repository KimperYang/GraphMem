import networkx as nx
from sentence_transformers import SentenceTransformer, util

class SemanticGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_node(self, node_name):
        embedding = self.model.encode(node_name)
        self.graph.add_node(node_name, embedding=embedding)
    
    def add_edge(self, node1, node2, relation):
        embedding = self.model.encode(relation)
        self.graph.add_edge(node1, node2, relation=relation, relation_embedding=embedding)
    
    def query(self, node1=None, node2=None, relation=None):
        query_node1_emb = self.model.encode(node1) if node1 else None
        query_node2_emb = self.model.encode(node2) if node2 else None
        query_relation_emb = self.model.encode(relation) if relation else None

        best_match = None
        best_score = -1

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

            # Update best match
            if score > best_score:
                best_score = score
                best_match = (n1, n2, attrs['relation'])

        return best_match, best_score

if __name__ == "__main__":
    g = SemanticGraph()
    g.add_node("Person A")
    g.add_node("Person B")
    g.add_node("Person C")

    g.add_edge("Person A", "Person B", "is a friend of")
    g.add_edge("Person B", "Person C", "works with")
    g.add_edge("Person A", "Person C", "is a family member of")

    query_result, score = g.query(node1="Person A", relation="friendship")
    print(f"Best Match: {query_result}, Score: {score:.4f}")

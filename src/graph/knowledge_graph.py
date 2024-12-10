import networkx as nx
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import pdb
import re

class SemanticKnowledgeGraph:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer(model_name)

    def add_node(self, input_node):

        print(input_node)
        
        if input_node[0] == "'" or input_node[0] == '"':
            pdb.set_trace()
            
        embedding = self.model.encode(input_node)

        if input_node not in self.graph.nodes:
            for node in self.graph.nodes:
                if util.cos_sim(embedding, self.graph.nodes[node]['embedding']).item() > 0.8:
                    return {
                        'node': node,
                        'message': f"Node '{input_node}' already exists with similar embedding"}
                         
            self.graph.add_node(input_node, embedding=embedding)
            
            return {
                'node': input_node,
                'message': f"Added node '{input_node}'"
            }
            
        else:
            return {
                'node': input_node,
                'message': f"Node '{input_node}' already exists"
            }
            
    def add_edge(self, node1, node2, relation, overwrite=False):
        quote_chars = "‘’“”\"'"
        node1 = node1.strip(quote_chars)
        node2 = node2.strip(quote_chars)
        relation = relation.strip(quote_chars)

        node1 = self.add_node(node1)['node']
        node2 = self.add_node(node2)['node']

        relation_embedding = self.model.encode(relation)

        if self.graph.has_edge(node1, node2):
            existing_relation = self.graph[node1][node2].get('relation', None)
            if overwrite:
                self.graph[node1][node2]['relation'] = relation
                self.graph[node1][node2]['relation_embedding'] = relation_embedding
                return {
                    'conflict': True,
                    'message': f"Edge ({node1}, {node2}) relation updated to '{relation}'",
                }

            return {
                'conflict': True,
                'message': (node1, existing_relation, node2),
            }

        else:
            self.graph.add_edge(node1, node2, relation=relation, relation_embedding=relation_embedding)
            return {
                'conflict': False,
                'message': f"Added edge ({node1}, {node2}) with relation '{relation}'",
            }

    def query(self, node1=None, node2=None, relation=None, top_k=1):
        assert (node1 is None) + (node2 is None) + (relation is None) == 1, "Only one element of the triplet can be None during query"

        matches = []
        scores = []

        if node1 is None:
            # Given node2 and relation, find node1
            if node2 in self.graph.nodes:
                query_node2_emb = self.graph.nodes[node2]['embedding']
            else:
                query_node2_emb = self.model.encode(node2)
            query_relation_emb = self.model.encode(relation)
            for n1 in self.graph.predecessors(node2):
                n1_emb = self.graph.nodes[n1]['embedding']
                rel_emb = self.graph[n1][node2]['relation_embedding']

                node_score = util.cos_sim(n1_emb, query_node2_emb).item()
                rel_score = util.cos_sim(rel_emb, query_relation_emb).item()
                total_score = node_score + rel_score

                matches.append(n1)
                scores.append(total_score)

            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            top_k_nodes = [(matches[i], scores[i]) for i in top_indices]
            return top_k_nodes

        elif node2 is None:
            # Given node1 and relation, find node2
            if node1 in self.graph.nodes:
                query_node1_emb = self.graph.nodes[node1]['embedding']
            else:
                query_node1_emb = self.model.encode(node1)
            query_relation_emb = self.model.encode(relation)
            for n2 in self.graph.successors(node1):
                n2_emb = self.graph.nodes[n2]['embedding']
                rel_emb = self.graph[node1][n2]['relation_embedding']

                node_score = util.cos_sim(n2_emb, query_node1_emb).item()
                rel_score = util.cos_sim(rel_emb, query_relation_emb).item()
                total_score = node_score + rel_score

                matches.append(n2)
                scores.append(total_score)

            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            top_k_nodes = [(matches[i], scores[i]) for i in top_indices]
            return top_k_nodes

        else:
            # Given node1 and node2, find relation
            if self.graph.has_edge(node1, node2):
                rel = self.graph[node1][node2]['relation']
                total_score = 2.0
                return [(rel, total_score)]
            else:
                return []
    
    def draw(self,file_name="graph.png"):
        pos = nx.shell_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=300, font_size=8)
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.savefig(file_name)
        #clear the plot
        plt.clf()
        
    
if __name__ == "__main__":
    kg = SemanticKnowledgeGraph()
    kg.add_edge("A", "B", "cooperate")
    kg.add_edge("B", "C", "compete")
    kg.add_edge("A", "C", "friend")
    kg.add_edge("A", "D", "enemy")

    # Query the relation between A and B
    relations = kg.query(node1="A", node2="B", relation=None)
    print("Relation between A and B:", relations)

    # Given node A and relation "cooperate", find nodes
    nodes = kg.query(node1="A", node2=None, relation="friend", top_k=2)
    print("Nodes that cooperate with A:", nodes)

    # Given node C and relation "compete", find starting nodes
    nodes = kg.query(node1=None, node2="C", relation="compete")
    print("Nodes that compete with C:", nodes)
    kg.draw()
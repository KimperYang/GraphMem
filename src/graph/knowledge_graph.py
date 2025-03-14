import networkx as nx
import numpy as np
import torch
from collections import deque
from sentence_transformers import SentenceTransformer, util
import difflib
import os
import random
import datetime
import matplotlib.pyplot as plt
import pdb
class SemanticKnowledgeGraph:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer(model_name)

    def _cos_sim(self, a, b):
        """
        计算余弦相似度，确保输入转换为 torch.float 类型
        """
        a_tensor = torch.tensor(a, dtype=torch.float)
        b_tensor = torch.tensor(b, dtype=torch.float)
        return util.cos_sim(a_tensor, b_tensor).item()

    def add_node(self, node):
        # 将节点文本编码为 embedding，并存储在图中
        emb = self.model.encode(node)
        if node not in self.graph.nodes:
            for n in self.graph.nodes:
                n_emb = self.graph.nodes[n]['embedding']
                sim = self._cos_sim(n_emb, emb)
                if sim > 0.8:
                    return {
                 'node': n,
                 'message': f"Node '{node}' already exists with similar embedding"}
                    
            self.graph.add_node(node, embedding=emb)
            return {
                 'node': node,
                 'message': f"Added node '{node}'"
             }
        else:
            return {
                 'node': node,
                 'message': f"Node '{node}' already exists"
             }

    def add_edge(self, node1, relation, node2, overwrite=False):
        quote_chars = "‘’“”\"'"
        node1 = node1.strip(quote_chars)
        node2 = node2.strip(quote_chars)
        relation = relation.strip(quote_chars)
        node1 = self.add_node(node1)['node']
        node2 = self.add_node(node2)['node']
        relation_emb = self.model.encode(relation)
        
        if self.graph.has_edge(node1, node2):
            existing_relation = self.graph[node1][node2].get('relation', None)
            if overwrite:
                self.graph[node1][node2]['relation'] = relation
                self.graph[node1][node2]['relation_embedding'] = relation_emb                 
                return {
                    'conflict': True,
                    'message': f"Edge ({node1}, {node2}) relation updated to '{relation}'",
                }
            return {
                'conflict': True,
                'message': f"Edge ({node1}, {node2}) already exists with relation '{existing_relation}'",
            }
            
        self.graph.add_edge(node1, node2, relation=relation, relation_embedding=relation_emb)
        return {
            'conflict': False,
            'message': f"Added edge ({node1}, {node2}) with relation '{relation}'",
        }
 

    def _apply_gcn(self, layers=2):
        nodes = list(self.graph.nodes)
        if not nodes:
            return
        n = len(nodes)
        index_map = {node: idx for idx, node in enumerate(nodes)}
        A = np.zeros((n, n), dtype=float)
        for u, v in self.graph.edges:
            if u in index_map and v in index_map:
                A[index_map[u], index_map[v]] = 1.0
        A_hat = A.copy()
        np.fill_diagonal(A_hat, 1.0)
        deg = A_hat.sum(axis=1)
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[deg == 0] = 0
        X = np.vstack([self.graph.nodes[node]['embedding'] for node in nodes])

        for _ in range(layers):
            X = deg_inv_sqrt[:, None] * (A_hat.dot(deg_inv_sqrt[:, None] * X))
            X = np.maximum(X, 0)

        for node in nodes:
            idx = index_map[node]
            self.graph.nodes[node]['embedding_gcn'] = X[idx]

    def query(self, node1=None, relation=None, node2=None, top_k=1, max_hops=1, use_gnn=False):
        """
        查询知识图谱。要求三元组 (node1, relation, node2) 中恰好有一个元素为 None，
        表示需要预测的部分。
        
        参数:
          - node1, relation, node2: 三元组的元素（其中一个必须为 None）
          - top_k: 返回的最佳匹配（或路径）的数量
          - max_hops: 多跳搜索时考虑的最大跳数
          - use_gnn: 若为 True，则在搜索前用 GCN 更新节点 embedding，提供多跳语义信息
        
        返回:
          - 如果缺少节点，返回符合要求的三元组列表；
          - 如果缺少关系（即寻找 node1 到 node2 的连接路径），返回路径（三元组链）的列表。
        """
        # 必须恰好只有一个元素为 None
        
        assert (node1 is None) + (node2 is None) + (relation is None) == 1, \
            "三元组中恰好只有一个元素可以为 None"

        matches = []
        scores = []

        # 如果使用 GNN，则先更新节点 embedding
        if use_gnn:
            self._apply_gcn(layers=max_hops)

        # 定义获取 embedding 的 key
        emb_key = 'embedding_gcn' if use_gnn else 'embedding'

        if node1 is None:
            # Given node2 and relation, find node1
            if node2 in self.graph.nodes:
                query_node2_emb = self.graph.nodes[node2][emb_key]
            else:
                best_matches = difflib.get_close_matches(node2, list(self.graph.nodes), n=1, cutoff=0.6)
                if best_matches:
                    node2 = best_matches[0]
                    query_node2_emb = self.graph.nodes[node2][emb_key]
                else:
                    # 若 difflib 没有匹配，再尝试用 embedding 语义相似度
                    query_target_emb = self.model.encode(node2)
                    best_match = None
                    best_score = -1
                    for n in self.graph.nodes:
                        n_emb = self.graph.nodes[n]['embedding']
                        sim = self._cos_sim(n_emb, query_target_emb)
                        if sim > best_score:
                            best_score = sim
                            best_match = n
                    node2 = best_match if best_match is not None else node2
                    query_node2_emb = self.graph.nodes[node2][emb_key]
                    
            query_relation_emb = self.model.encode(relation)
            
            for n1 in self.graph:
                if n1 == node2:
                    continue
                
                n1_emb = self.graph.nodes[n1][emb_key]
                if self.graph.has_edge(n1, node2):
                    rel_emb = self.graph[n1][node2]['relation_embedding']
                    node_score = self._cos_sim(n1_emb, query_node2_emb)
                    rel_score = self._cos_sim(rel_emb, query_relation_emb)
                    total_score = node_score + rel_score
                    matches.append((n1, self.graph[n1][node2]['relation'], node2))
                    scores.append(total_score)
            
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            top_k_nodes = [matches[i] for i in top_indices]
            return top_k_nodes


        elif node2 is None:
            # Given node1 and relation, find node2
            if node1 in self.graph.nodes:
                query_node1_emb = self.graph.nodes[node1][emb_key]
            else:
                best_matches = difflib.get_close_matches(node1, list(self.graph.nodes), n=1, cutoff=0.6)
                if best_matches:
                    node1 = best_matches[0]
                    query_node1_emb = self.graph.nodes[node1][emb_key]
                else:
                    # 若 difflib 没有匹配，再尝试用 embedding 语义相似度
                    query_target_emb = self.model.encode(node1)
                    best_match = None
                    best_score = -1
                    for n in self.graph.nodes:
                        n_emb = self.graph.nodes[n]['embedding']
                        sim = self._cos_sim(n_emb, query_target_emb)
                        if sim > best_score:
                            best_score = sim
                            best_match = n
                    node1 = best_match if best_match is not None else node1
                    query_node1_emb = self.graph.nodes[node1][emb_key]
            query_relation_emb = self.model.encode(relation)
            
            for n2 in self.graph:
                if n2 == node1:
                    continue
                n2_emb = self.graph.nodes[n2][emb_key]
                if self.graph.has_edge(node1, n2):
                    rel_emb = self.graph[node1][n2]['relation_embedding']
                    node_score = self._cos_sim(n2_emb, query_node1_emb)
                    rel_score = self._cos_sim(rel_emb, query_relation_emb)
                    total_score = node_score + rel_score
                    matches.append((node1, self.graph[node1][n2]['relation'], n2))
                    scores.append(total_score)
                    
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            top_k_nodes = [matches[i] for i in top_indices]
            
            return top_k_nodes

        # Case 3: 缺少关系，即寻找 node1 到 node2 的连接路径
        else:
            # 如果目标节点不存在，先通过 difflib 根据字符串相似性找到最接近的节点
            if node1 not in self.graph.nodes:
                best_matches = difflib.get_close_matches(node1, list(self.graph.nodes), n=1, cutoff=0.6)
                if best_matches:
                    node1 = best_matches[0]
                else:
                    # 若 difflib 没有匹配，再尝试用 embedding 语义相似度
                    query_target_emb = self.model.encode(node1)
                    best_match = None
                    best_score = -1
                    for n in self.graph.nodes:
                        n_emb = self.graph.nodes[n]['embedding']
                        sim = self._cos_sim(n_emb, query_target_emb)
                        if sim > best_score:
                            best_score = sim
                            best_match = n
                    node1 = best_match if best_match is not None else node1
                    
                    
            if node2 not in self.graph.nodes:
                best_matches = difflib.get_close_matches(node2, list(self.graph.nodes), n=1, cutoff=0.6)
                if best_matches:
                    node2 = best_matches[0]
                else:
                    # 若 difflib 没有匹配，再尝试用 embedding 语义相似度
                    query_target_emb = self.model.encode(node2)
                    best_match = None
                    best_score = -1
                    for n in self.graph.nodes:
                        n_emb = self.graph.nodes[n]['embedding']
                        sim = self._cos_sim(n_emb, query_target_emb)
                        if sim > best_score:
                            best_score = sim
                            best_match = n
                    node2 = best_match if best_match is not None else node2
                    
                    
            # 如果直接边存在，则直接返回
            if max_hops <= 1:
                if self.graph.has_edge(node1, node2):
                    rel = self.graph[node1][node2]['relation']
                    return [(node1, rel, node2)]
                else:
                    return []

            # 使用 beam search 探索多跳路径，并利用 embedding 计算语义相似度
            target_emb = self.graph.nodes[node2][emb_key]
            # 初始化候选路径，每个元素为 (path, 累计得分)
            candidates = [([node1], 0.0)]
            beam_width = top_k * 5  # 设置 beam 宽度
            final_paths = []
            for hop in range(max_hops):
                new_candidates = []
                for path, score in candidates:
                    last_node = path[-1]
                    for neighbor in self.graph.successors(last_node):
                        if neighbor in path:  # 避免环路
                            continue
                        new_path = path + [neighbor]
                        neighbor_emb = self.graph.nodes[neighbor][emb_key]
                        # 计算当前 neighbor 与目标节点之间的余弦相似度
                        sim = self._cos_sim(neighbor_emb, target_emb)
                        new_score = score + sim
                        new_candidates.append((new_path, new_score))
                if not new_candidates:
                    break
                new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                for path, score in new_candidates:
                    if path[-1] == node2:
                        final_paths.append((path, score))
                if final_paths:
                    break
                candidates = new_candidates

            if not final_paths:
                return []
            
            final_paths = sorted(final_paths, key=lambda x: x[1], reverse=True)[:top_k]
            results = []
            for path, score in final_paths:
                triple_path = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    triple_path.append((u, self.graph[u][v]['relation'], v))
                results.append(triple_path)
            return results
        
    def draw(self,file_name="graph.png"):
        
         pos = nx.spring_layout(self.graph)
         plt.figure(figsize=(12, 12))
         
         node_colors = [f'#{random.randint(0xAAAAAA, 0xFFFFFF):06x}' for _ in range(len(self.graph.nodes))]
         
         nx.draw(self.graph, pos, with_labels=True, node_size=600, font_size=16, node_color=node_colors)
         edge_labels = nx.get_edge_attributes(self.graph, 'relation')
         nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
 
         current_time = datetime.datetime.now()
         time_str = current_time.strftime("%Y%m%d-%H%M%S")
 
         dump_path = "figs/" + time_str + "_" + file_name
         if not os.path.exists("figs"):
             os.makedirs("figs")
         print(f"Graph picture dumped to {dump_path}")
         plt.savefig(dump_path)
         
def main():
    # 创建知识图谱实例
    kg = SemanticKnowledgeGraph()

    # 添加节点
    for node in ["Alice", "Bob", "Charlie", "David"]:
        kg.add_node(node)

    # 添加边和对应关系
    kg.add_edge("Alice", "Bob", "friend")
    kg.add_edge("Bob", "Charlie", "colleague")
    kg.add_edge("Charlie", "David", "neighbor")
    kg.add_edge("Alice", "David", "relative")

    print("【直接边查询】 (Alice -> David):")
    result = kg.query(node1="Alice", node2="David", relation=None, max_hops=1)
    print(result)

    print("\n【多跳查询】 (Alice -> Charlie, max_hops=2) 不使用 GCN:")
    result = kg.query(node1="Alice", node2="Charlie", relation=None, max_hops=2, use_gnn=False)
    print(result)

    print("\n【多跳查询】 (Alice -> Charlie, max_hops=2) 使用 GCN 和语义相似度 (beam search):")
    result = kg.query(node1="Alice", node2="Charlie", relation=None, max_hops=2, use_gnn=True, top_k=2)
    print(result)

    # 语义相似性测试：使用拼写略有不同的目标 "Charly"
    print("\n【语义相似性测试】 多跳查询 (Alice -> 'Charly', max_hops=2) 使用 GCN:")
    result = kg.query(node1="Alice", node2="Charly", relation=None, max_hops=2, use_gnn=True, top_k=2)
    print(result)

    print("\n【缺失节点查询】 查询缺失节点 (Alice, friend, ?):")
    result = kg.query(node1="Alice", node2=None, relation="friend")
    print(result)

    print("\n【缺失节点查询】 查询缺失节点 (?, colleague, 'Charly'):")
    result = kg.query(node1=None, node2="Charly", relation="colleague")
    print(result)

if __name__ == "__main__":
    main()

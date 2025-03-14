import networkx as nx
import numpy as np
import torch
from collections import deque
from sentence_transformers import SentenceTransformer, util
import difflib
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
        self.graph.add_node(node, embedding=emb)

    def add_edge(self, u, v, relation):
        # 将关系文本编码为 embedding，并存储边信息
        rel_emb = self.model.encode(relation)
        self.graph.add_edge(u, v, relation=relation, relation_embedding=rel_emb)

    def _apply_gcn(self, layers=2):
        """
        应用简单的图卷积网络（GCN）更新，将每个节点的 embedding 
        用归一化的邻居特征聚合更新，传播多跳邻域信息。
        """
        nodes = list(self.graph.nodes)
        if not nodes:
            return
        n = len(nodes)
        index_map = {node: idx for idx, node in enumerate(nodes)}
        # 构建图的邻接矩阵 A
        A = np.zeros((n, n), dtype=float)
        for u, v in self.graph.edges:
            if u in index_map and v in index_map:
                A[index_map[u], index_map[v]] = 1.0
        # 加上自环（A_hat = A + I）
        A_hat = A.copy()
        np.fill_diagonal(A_hat, 1.0)
        # 计算 D^(-1/2)
        deg = A_hat.sum(axis=1)
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[deg == 0] = 0
        # 构建特征矩阵 X（每一行为一个节点的 embedding）
        X = np.vstack([self.graph.nodes[node]['embedding'] for node in nodes])
        # GCN 传播，每一层: X <- D^(-1/2) * A_hat * D^(-1/2) * X, 后接 ReLU 激活
        for _ in range(layers):
            X = deg_inv_sqrt[:, None] * (A_hat.dot(deg_inv_sqrt[:, None] * X))
            X = np.maximum(X, 0)
        # 将更新后的 embedding 存储到节点属性 'embedding_gcn' 中
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

        # Case 1: 缺少 node1，给定 ( ?, relation, node2 )
        if node1 is None:
            if node2 in self.graph.nodes:
                query_node2_emb = self.graph.nodes[node2][emb_key]
            else:
                query_node2_emb = self.model.encode(node2)
                best_match = None
                best_score = -1
                for n in self.graph.nodes:
                    n_emb = self.graph.nodes[n][emb_key]
                    sim = self._cos_sim(n_emb, query_node2_emb)
                    if sim > best_score:
                        best_score = sim
                        best_match = n
                node2 = best_match if best_match is not None else node2
                if best_match:
                    query_node2_emb = self.graph.nodes[node2][emb_key]
            query_rel_emb = self.model.encode(relation)
            for n1 in self.graph.nodes:
                if n1 == node2:
                    continue
                if self.graph.has_edge(n1, node2):
                    rel = self.graph[n1][node2]['relation']
                    rel_emb = self.graph[n1][node2]['relation_embedding']
                    rel_score = self._cos_sim(rel_emb, query_rel_emb)
                    matches.append((n1, rel, node2))
                    scores.append(rel_score)
            top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [matches[i] for i in top_k_idx]

        # Case 2: 缺少 node2，给定 ( node1, relation, ? )
        elif node2 is None:
            if node1 in self.graph.nodes:
                query_node1_emb = self.graph.nodes[node1][emb_key]
            else:
                query_node1_emb = self.model.encode(node1)
                best_match = None
                best_score = -1
                for n in self.graph.nodes:
                    n_emb = self.graph.nodes[n][emb_key]
                    sim = self._cos_sim(n_emb, query_node1_emb)
                    if sim > best_score:
                        best_score = sim
                        best_match = n
                node1 = best_match if best_match is not None else node1
                if best_match:
                    query_node1_emb = self.graph.nodes[node1][emb_key]
            query_rel_emb = self.model.encode(relation)
            for n2_candidate in self.graph.nodes:
                if n2_candidate == node1:
                    continue
                if self.graph.has_edge(node1, n2_candidate):
                    rel = self.graph[node1][n2_candidate]['relation']
                    rel_emb = self.graph[node1][n2_candidate]['relation_embedding']
                    node_score = self._cos_sim(self.graph.nodes[n2_candidate][emb_key], query_node1_emb)
                    rel_score = self._cos_sim(rel_emb, query_rel_emb)
                    total_score = node_score + rel_score
                    matches.append((node1, rel, n2_candidate))
                    scores.append(total_score)
            top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [matches[i] for i in top_k_idx]

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
                        n_emb = self.graph.nodes[n][emb_key]
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
                        n_emb = self.graph.nodes[n][emb_key]
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

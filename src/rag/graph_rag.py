"""
グラフ構造を活用したRAGシステム
"""

from typing import List, Dict, Optional, Union, Tuple, Set
from pathlib import Path
import numpy as np
import re
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch.nn.functional as F
from ..knowledge_graph.neo4j_manager import Neo4jManager
from ..utils.device import get_device, move_to_device

class ESGGraphRAG:
    """ESGドメインのGraphRAGシステム"""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        llm_model_name: str = "rinna/japanese-gpt-neox-small",
        embedding_model_name: str = "sonoisa/sentence-bert-base-ja-mean-tokens",
        device: Optional[torch.device] = None,
        use_mps: bool = False
    ):
        """
        初期化
        
        Args:
            neo4j_uri: Neo4jのURI
            neo4j_user: Neo4jのユーザー名
            neo4j_password: Neo4jのパスワード
            llm_model_name: 使用するLLMモデル名
            embedding_model_name: 使用するエンベッディングモデル名
            device: 使用するデバイス
            use_mps: MPSデバイスを使用するかどうか
        """
        self.device = device if device is not None else get_device()
        self.use_mps = use_mps
        
        # Neo4jマネージャーの初期化
        self.neo4j = Neo4jManager(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
        
        # LLMの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        if self.use_mps:
            self.llm = self.llm.to('mps')
        else:
            self.llm = move_to_device(self.llm, self.device)
        
        # エンベッディングモデルの初期化（sentence-transformersの代わりにtransformersを直接使用）
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        if self.use_mps:
            self.embedding_model = self.embedding_model.to('mps')
        else:
            self.embedding_model = move_to_device(self.embedding_model, self.device)
        
        # ノード埋め込みの保存用
        self.node_embeddings = {}
    
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """テキストの埋め込みを取得"""
        # バッチ処理のための準備
        encoded = self.embedding_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        if self.use_mps:
            encoded = {k: v.to('mps') for k, v in encoded.items()}
        else:
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # モデルの出力を取得
        with torch.no_grad():
            outputs = self.embedding_model(**encoded)
            # [CLS]トークンの出力を使用
            embeddings = outputs.last_hidden_state[:, 0, :]
            # L2正規化
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def search_relevant_subgraph(
        self,
        query: str,
        max_nodes: int = 10,
        max_depth: int = 2,
        similarity_threshold: float = 0.3,
        category_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        クエリに関連するサブグラフを検索
        
        Args:
            query: 検索クエリ
            max_nodes: 取得する最大ノード数
            max_depth: 探索する最大の深さ
            similarity_threshold: 類似度の閾値
            category_weights: カテゴリごとの重み付け
            
        Returns:
            関連するサブグラフ情報
        """
        # クエリの埋め込みを計算
        query_embedding = self._get_embeddings([query])[0]
        
        # デフォルトのカテゴリ重み
        if category_weights is None:
            category_weights = {
                "Environment": 1.0,
                "Social": 1.0,
                "Governance": 1.0,
                "Other": 0.8
            }
        
        # Neo4jでの検索クエリ
        cypher_query = """
        MATCH (n:Concept)
        RETURN n.name AS name, n.category AS category, n.description AS description
        """
        
        with self.neo4j.driver.session() as session:
            result = session.run(cypher_query)
            nodes = [(record["name"], record["category"], record.get("description", "")) for record in result]
        
        # ノードの類似度を計算
        similarities = []
        for name, category, description in nodes:
            if name not in self.node_embeddings:
                # 埋め込みがない場合は計算
                text = f"{name} ({category}). {description}"
                embedding = self._get_embeddings([text])[0]
                self.node_embeddings[name] = embedding
            else:
                embedding = self.node_embeddings[name]
            
            # コサイン類似度を計算
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                embedding.unsqueeze(0)
            ).item()
            
            # カテゴリの重みを適用
            weight = category_weights.get(category, 1.0)
            weighted_similarity = similarity * weight
            
            similarities.append((name, weighted_similarity))
        
        # 類似度でソート
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 閾値以上の類似度を持つノードを選択
        relevant_nodes = [
            name for name, sim in similarities
            if sim >= similarity_threshold
        ][:max_nodes]
        
        if not relevant_nodes:
            return {"nodes": [], "relationships": [], "statistics": {
                "total_nodes": 0,
                "total_relationships": 0,
                "category_distribution": {}
            }}
        
        # 関連ノード間のパスを取得
        paths_query = """
        MATCH path = (n:Concept)-[r:ESG_RELATION*1..%d]->(m:Concept)
        WHERE n.name IN $relevant_nodes AND m.name IN $relevant_nodes
        RETURN path
        """ % max_depth
        
        with self.neo4j.driver.session() as session:
            result = session.run(paths_query, relevant_nodes=relevant_nodes)
            paths = list(result)
        
        # サブグラフの構築
        nodes_set = set()
        relationships_set = set()
        
        for path in paths:
            path_nodes = path["path"].nodes
            path_relationships = path["path"].relationships
            
            for node in path_nodes:
                nodes_set.add((
                    node["name"],
                    node.get("category", "Other"),
                    node.get("description", "")
                ))
            
            for rel in path_relationships:
                relationships_set.add((
                    rel.start_node["name"],
                    rel.type,
                    rel.end_node["name"],
                    rel.get("confidence", 1.0)
                ))
        
        # 結果の整形
        nodes = [
            {
                "name": name,
                "category": category,
                "description": description
            }
            for name, category, description in nodes_set
        ]
        
        relationships = [
            {
                "source": source,
                "type": rel_type,
                "target": target,
                "confidence": confidence
            }
            for source, rel_type, target, confidence in relationships_set
        ]
        
        # カテゴリ分布の計算
        category_counts = {}
        for node in nodes:
            category = node["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "nodes": nodes,
            "relationships": relationships,
            "statistics": {
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "category_distribution": category_counts
            }
        }
    
    def generate_response(
        self,
        query: str,
        subgraph: Dict,
        max_length: int = 512
    ) -> str:
        """
        サブグラフを考慮して回答を生成
        
        Args:
            query: ユーザーのクエリ
            subgraph: 関連するサブグラフ情報
            max_length: 生成する最大文字数
            
        Returns:
            生成された回答
        """
        # コンテキストの構築
        context = self._build_context_from_subgraph(subgraph)
        
        # 簡易的な回答生成（実際のシステムではLLMを使用）
        response = f"クエリ「{query}」に関連する情報:\n\n"
        
        # カテゴリごとの情報を追加
        for category, concepts in context.items():
            response += f"\n{category}に関する情報:\n"
            for concept, relations in concepts.items():
                response += f"- {concept}: {', '.join(relations)}\n"
        
        # 最大長に制限
        return response[:max_length]
    
    def _build_context_from_subgraph(self, subgraph: Dict) -> Dict[str, Dict[str, List[str]]]:
        """サブグラフから文脈情報を構築"""
        context = {}
        
        # ノードをカテゴリごとに整理
        for node in subgraph["nodes"]:
            category = node.get("category", "Other")
            if category not in context:
                context[category] = {}
            context[category][node["name"]] = []
        
        # 関係の情報を追加
        for rel in subgraph["relationships"]:
            source_category = None
            for node in subgraph["nodes"]:
                if node["name"] == rel["source"]:
                    source_category = node.get("category", "Other")
                    break
            
            if source_category and rel["source"] in context[source_category]:
                context[source_category][rel["source"]].append(
                    f"{rel['type']} -> {rel['target']}"
                )
        
        return context
    
    def _tokenize(self, text: str) -> List[str]:
        """テキストの分かち書き（簡易版）"""
        # 記号を除去し、空白で分割
        text = re.sub(r'[、。！？「」『』（）［］\s]', ' ', text)
        return [word for word in text.split() if word]
    
    def close(self):
        """リソースの解放"""
        self.neo4j.close()
    
    def update_node_embeddings(self, batch_size: int = 32):
        """
        全ノードの埋め込みを更新
        
        Args:
            batch_size: バッチサイズ
        """
        # Neo4jから全ノードを取得
        cypher_query = """
        MATCH (n:Concept)
        RETURN n.name AS name, n.category AS category
        """
        
        with self.neo4j.driver.session() as session:
            result = session.run(cypher_query)
            nodes = [(record["name"], record["category"]) for record in result]
        
        # バッチ処理でノード埋め込みを計算
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            texts = [f"{name} ({category})" for name, category in batch]
            
            embeddings = self._get_embeddings(texts)
            
            # 埋め込みを保存
            for j, (name, _) in enumerate(batch):
                self.node_embeddings[name] = embeddings[j]
        
        print(f"ノード埋め込みを更新しました（{len(self.node_embeddings)}ノード）") 
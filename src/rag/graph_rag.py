"""
グラフ構造を活用したRAGシステム
"""

from typing import List, Dict, Optional, Union, Tuple, Set
from pathlib import Path
import numpy as np
import re
from collections import Counter
from ..knowledge_graph.neo4j_manager import Neo4jManager

class ESGGraphRAG:
    """ESGドメインのGraphRAGシステム"""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            neo4j_uri: Neo4jのURI
            neo4j_user: Neo4jのユーザー名
            neo4j_password: Neo4jのパスワード
        """
        # Neo4jマネージャーの初期化
        self.neo4j = Neo4jManager(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
    
    def search_relevant_subgraph(
        self,
        query: str,
        max_nodes: int = 10,
        max_depth: int = 2,
        similarity_threshold: float = 0.5,
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
        # デフォルトのカテゴリ重み
        if category_weights is None:
            category_weights = {
                "Environment": 1.0,
                "Social": 1.0,
                "Governance": 1.0
            }
        
        # クエリの単語を抽出
        query_words = set(self._tokenize(query))
        
        # Neo4jでの検索クエリ
        cypher_query = """
        // ステップ1: クエリに関連するノードの検索
        MATCH (n:Concept)
        WHERE any(word IN $query_words WHERE n.name CONTAINS word)
        WITH n, n.category AS category
        
        // ステップ2: カテゴリによる重み付け
        WITH n, CASE category
            WHEN 'Environment' THEN $env_weight
            WHEN 'Social' THEN $social_weight
            WHEN 'Governance' THEN $gov_weight
            ELSE 1.0
        END AS weight
        ORDER BY weight DESC
        LIMIT $max_initial_nodes
        
        // ステップ3: 関連するパスの探索
        MATCH path = (n)-[r:ESG_RELATION*1..%d]->(m)
        WHERE ALL(rel IN r WHERE rel.confidence IS NULL OR rel.confidence >= 0.5)
        WITH COLLECT(path) AS paths
        
        // ステップ4: パスの展開とユニークなノードと関係の抽出
        UNWIND paths AS p
        WITH DISTINCT nodes(p) AS nodes, relationships(p) AS rels
        
        // ステップ5: 結果の整形
        RETURN 
            [n IN nodes | {
                id: id(n),
                name: n.name,
                category: n.category,
                properties: properties(n)
            }] AS nodes,
            [r IN rels | {
                source: startNode(r).name,
                type: r.type,
                target: endNode(r).name,
                properties: properties(r)
            }] AS relationships
        LIMIT %d
        """ % (max_depth, max_nodes)
        
        # パラメータの準備
        params = {
            "query_words": list(query_words),
            "max_initial_nodes": max_nodes,
            "env_weight": category_weights.get("Environment", 1.0),
            "social_weight": category_weights.get("Social", 1.0),
            "gov_weight": category_weights.get("Governance", 1.0)
        }
        
        # クエリの実行
        with self.neo4j.driver.session() as session:
            result = session.run(cypher_query, params)
            subgraph = result.single()
            
            if not subgraph:
                return {"nodes": [], "relationships": []}
        
        # 結果の後処理
        processed_result = self._process_subgraph_result(subgraph)
        
        return processed_result
    
    def _process_subgraph_result(self, subgraph: Dict) -> Dict:
        """サブグラフの結果を後処理"""
        # ノードの重複除去とソート
        unique_nodes = {}
        for node in subgraph["nodes"]:
            if node["name"] not in unique_nodes:
                unique_nodes[node["name"]] = node
        
        # 関係の重複除去とソート
        unique_relations = {}
        for rel in subgraph["relationships"]:
            key = f"{rel['source']}-{rel['type']}-{rel['target']}"
            if key not in unique_relations:
                unique_relations[key] = rel
        
        # カテゴリごとのノード数を計算
        category_counts = {}
        for node in unique_nodes.values():
            category = node.get("category", "Other")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "nodes": list(unique_nodes.values()),
            "relationships": list(unique_relations.values()),
            "statistics": {
                "total_nodes": len(unique_nodes),
                "total_relationships": len(unique_relations),
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
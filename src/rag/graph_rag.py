#!/usr/bin/env python3
"""
ESG GraphRAGシステム
"""

from typing import Dict, List, Optional
from openai import OpenAI
import torch
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import json

class ESGGraphRAG:
    """ESG GraphRAGシステム"""
    
    def __init__(
        self,
        openai_api_key: str,
        embedding_model_name: str,
        device: torch.device,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str
    ):
        """
        初期化
        
        Args:
            openai_api_key: OpenAI APIキー
            embedding_model_name: 文埋め込みモデル名
            device: 計算デバイス
            neo4j_uri: Neo4jのURI
            neo4j_user: Neo4jのユーザー名
            neo4j_password: Neo4jのパスワード
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = SentenceTransformer(embedding_model_name).to(device)
        self.device = device
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # ノード埋め込みのキャッシュ
        self.node_embeddings = {}
    
    def update_node_embeddings(self, batch_size: int = 32):
        """
        全ノードの埋め込みを更新
        
        Args:
            batch_size: バッチサイズ
        """
        with self.driver.session() as session:
            # 全ノードの取得
            result = session.run("MATCH (n) RETURN n.name AS name")
            nodes = [record["name"] for record in result]
            
            # バッチ処理
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                embeddings = self.embedding_model.encode(
                    batch,
                    convert_to_tensor=True,
                    device=self.device
                )
                
                # キャッシュの更新
                for node, embedding in zip(batch, embeddings):
                    self.node_embeddings[node] = embedding
    
    def search_relevant_subgraph(
        self,
        query: str,
        max_nodes: int = 10,
        max_depth: int = 2
    ) -> Dict:
        """
        クエリに関連するサブグラフを検索
        
        Args:
            query: 検索クエリ
            max_nodes: 最大ノード数
            max_depth: 最大探索深度
            
        Returns:
            サブグラフ（ノードと関係のリスト）
        """
        # クエリの埋め込み
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )
        
        # 類似度の計算
        similarities = {}
        for node, embedding in self.node_embeddings.items():
            similarity = torch.cosine_similarity(query_embedding, embedding, dim=0)
            similarities[node] = similarity.item()
        
        # 類似度の高いノードを選択
        seed_nodes = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_nodes // 2]
        seed_nodes = [node for node, _ in seed_nodes]
        
        # サブグラフの取得
        with self.driver.session() as session:
            # シードノードを起点に探索
            cypher_query = f"""
            MATCH path = (n)-[*0..{max_depth}]-(m)
            WHERE n.name IN $seeds
            WITH DISTINCT nodes(path) AS nodes, relationships(path) AS rels
            UNWIND nodes AS node
            WITH COLLECT(DISTINCT {{
                name: node.name,
                category: node.category
            }}) AS nodes,
            rels
            UNWIND rels AS rel
            WITH nodes, COLLECT(DISTINCT {{
                source: startNode(rel).name,
                target: endNode(rel).name,
                type: type(rel)
            }}) AS relationships
            RETURN nodes, relationships
            """
            
            result = session.run(
                cypher_query,
                seeds=seed_nodes
            )
            
            record = result.single()
            if record is None:
                return {"nodes": [], "relationships": []}
            
            return {
                "nodes": record["nodes"][:max_nodes],
                "relationships": record["relationships"]
            }
    
    def generate_response(
        self,
        query: str,
        subgraph: Dict,
        temperature: float = 0.7
    ) -> Dict:
        """
        クエリに対する回答を生成
        
        Args:
            query: 質問
            subgraph: 関連するサブグラフ
            temperature: 生成の温度
            
        Returns:
            構造化された回答
        """
        # コンテキストの構築
        context = "以下のESGに関する知識グラフの情報を参考に回答してください：\n\n"
        
        # ノードの情報
        context += "【概念】\n"
        for node in subgraph["nodes"]:
            context += f"- {node['name']} (カテゴリ: {node['category']})\n"
        
        # 関係の情報
        context += "\n【関係性】\n"
        for rel in subgraph["relationships"]:
            context += f"- {rel['source']} → {rel['type']} → {rel['target']}\n"
        
        # プロンプトの構築
        prompt = f"""
{context}

質問: {query}

以下の形式で回答を構造化してJSON形式で出力してください：

{{
    "overview": "概要説明",
    "key_initiatives": [
        {{
            "title": "施策のタイトル",
            "description": "施策の説明",
            "implementation": "実施方法"
        }}
    ],
    "challenges": [
        "課題1",
        "課題2"
    ],
    "metrics": [
        {{
            "name": "指標名",
            "target": "目標値",
            "timeline": "達成期間"
        }}
    ],
    "conclusion": {{
        "summary": "まとめ",
        "future_outlook": "今後の展望"
    }},
    "references": [
        {{
            "concept": "参照した概念名",
            "category": "概念のカテゴリ",
            "relevance": "関連性の説明"
        }}
    ]
}}

注意：
- 回答は必ず上記のJSON形式で出力してください
- 各セクションは与えられた知識グラフの情報に基づいて具体的に記述してください
- 施策は3-5個程度、課題は2-3個程度、指標は2-3個程度を目安に出力してください
"""

        # OpenAI APIを使用して回答を生成
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "あなたはESGの専門家です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        # JSON形式の回答をパース
        return json.loads(response.choices[0].message.content)
    
    def close(self):
        """リソースの解放"""
        self.driver.close() 
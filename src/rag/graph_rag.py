"""
グラフ構造を活用したRAGシステム
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from ..utils.device import get_device, move_to_device
from ..knowledge_graph.neo4j_manager import Neo4jManager

class ESGGraphRAG:
    """ESGドメインのGraphRAGシステム"""
    
    def __init__(
        self,
        llm_model_name: str = "rinna/japanese-gpt-neox-small",
        embedding_model_name: str = "sonoisa/sentence-bert-base-ja-mean-tokens",
        device: Optional[torch.device] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            llm_model_name: 使用するLLMモデル名
            embedding_model_name: 文埋め込みモデル名
            device: 使用するデバイス
            neo4j_uri: Neo4jのURI
            neo4j_user: Neo4jのユーザー名
            neo4j_password: Neo4jのパスワード
        """
        self.device = device if device is not None else get_device()
        
        # LLMの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.llm = move_to_device(self.llm, self.device)
        
        # 文埋め込みモデルの初期化
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model = move_to_device(self.embedding_model, self.device)
        
        # Neo4jマネージャーの初期化
        self.neo4j = Neo4jManager(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
        
        # 推論モードに設定
        self.llm.eval()
        self.embedding_model.eval()
    
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
            max_nodes: 取得する最大ノード数
            max_depth: 探索する最大の深さ
            
        Returns:
            関連するサブグラフ情報
        """
        # クエリの埋め込み
        query_embedding = self.compute_embedding(query)
        
        # Neo4jでの類似度検索クエリ
        cypher_query = """
        MATCH (n:Concept)
        WHERE n.embedding IS NOT NULL
        WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS score
        ORDER BY score DESC
        LIMIT 1
        MATCH path = (n)-[r:ESG_RELATION*1..%d]->(m)
        WITH COLLECT(path) AS paths
        UNWIND paths AS p
        WITH DISTINCT nodes(p) AS nodes, relationships(p) AS rels
        RETURN 
            [n IN nodes | {id: id(n), name: n.name, category: n.category}] AS nodes,
            [r IN rels | {
                source: startNode(r).name,
                type: r.type,
                target: endNode(r).name,
                properties: properties(r)
            }] AS relationships
        LIMIT %d
        """ % (max_depth, max_nodes)
        
        with self.neo4j.driver.session() as session:
            result = session.run(
                cypher_query,
                query_embedding=query_embedding.tolist()
            )
            subgraph = result.single()
        
        return {
            "nodes": subgraph["nodes"],
            "relationships": subgraph["relationships"]
        }
    
    def generate_response(
        self,
        query: str,
        subgraph: Dict,
        max_length: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        サブグラフを考慮して回答を生成
        
        Args:
            query: ユーザーのクエリ
            subgraph: 関連するサブグラフ情報
            max_length: 生成する最大トークン数
            temperature: 生成の温度パラメータ
            
        Returns:
            生成された回答
        """
        # コンテキストの構築
        context = self._build_context_from_subgraph(subgraph)
        
        # プロンプトの構築
        prompt = f"""以下の情報を参考に、質問に対して正確に回答してください。

参考情報：
{context}

質問：{query}

回答："""
        
        # LLMによる回答生成
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("回答：")[-1].strip()
    
    def _build_context_from_subgraph(self, subgraph: Dict) -> str:
        """サブグラフから文脈情報を構築"""
        context = []
        
        # ノードの情報を追加
        nodes_by_category = {}
        for node in subgraph["nodes"]:
            category = node.get("category", "Other")
            if category not in nodes_by_category:
                nodes_by_category[category] = []
            nodes_by_category[category].append(node["name"])
        
        for category, nodes in nodes_by_category.items():
            context.append(f"{category}カテゴリの概念：{', '.join(nodes)}")
        
        # 関係の情報を追加
        for rel in subgraph["relationships"]:
            context.append(
                f"{rel['source']}は{rel['target']}と{rel['type']}の関係があります。"
            )
            if "description" in rel.get("properties", {}):
                context.append(f"説明: {rel['properties']['description']}")
        
        return "\n".join(context)
    
    @torch.no_grad()
    def compute_embedding(self, text: str) -> torch.Tensor:
        """テキストの埋め込みを計算"""
        return self.embedding_model.encode(
            text,
            convert_to_tensor=True,
            device=self.device
        )
    
    def update_node_embeddings(self, batch_size: int = 32) -> None:
        """
        Neo4jのノード埋め込みを更新
        
        Args:
            batch_size: バッチサイズ
        """
        # 全ノードの取得
        with self.neo4j.driver.session() as session:
            result = session.run("MATCH (n:Concept) RETURN n.name AS name")
            nodes = [record["name"] for record in result]
        
        # バッチ処理で埋め込みを計算
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch)
            
            # Neo4jの更新
            with self.neo4j.driver.session() as session:
                for name, embedding in zip(batch, embeddings):
                    session.run(
                        """
                        MATCH (n:Concept {name: $name})
                        SET n.embedding = $embedding
                        """,
                        name=name,
                        embedding=embedding.tolist()
                    )
    
    def close(self):
        """リソースの解放"""
        self.neo4j.close() 
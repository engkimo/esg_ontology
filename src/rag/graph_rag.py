"""
グラフ構造を活用したRAGシステム
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Union, Tuple, Set
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
        
        # トークナイザーの設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
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
        
        # クエリの埋め込み
        query_embedding = self.compute_embedding(query)
        
        # Neo4jでの類似度検索と重み付けクエリ
        cypher_query = """
        // ステップ1: 類似度に基づく初期ノードの検索
        MATCH (n:Concept)
        WHERE n.embedding IS NOT NULL
        WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS similarity
        WHERE similarity >= $similarity_threshold
        
        // ステップ2: カテゴリによる重み付け
        WITH n, similarity * CASE n.category
            WHEN 'Environment' THEN $env_weight
            WHEN 'Social' THEN $social_weight
            WHEN 'Governance' THEN $gov_weight
            ELSE 1.0
        END AS weighted_score
        ORDER BY weighted_score DESC
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
            "query_embedding": query_embedding.tolist(),
            "similarity_threshold": similarity_threshold,
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
    
    def get_path_between_concepts(
        self,
        source: str,
        target: str,
        max_depth: int = 3
    ) -> List[Dict]:
        """
        2つの概念間のパスを取得
        
        Args:
            source: 開始ノード
            target: 終了ノード
            max_depth: 最大深さ
            
        Returns:
            パス情報のリスト
        """
        cypher_query = """
        MATCH path = shortestPath((s:Concept {name: $source})-[r:ESG_RELATION*1..%d]->(t:Concept {name: $target}))
        RETURN [n IN nodes(path) | n.name] AS nodes,
               [r IN relationships(path) | r.type] AS relations
        """ % max_depth
        
        with self.neo4j.driver.session() as session:
            result = session.run(cypher_query, source=source, target=target)
            paths = []
            for record in result:
                paths.append({
                    "nodes": record["nodes"],
                    "relations": record["relations"]
                })
            
            return paths
    
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

    def hybrid_search(
        self,
        query: str,
        text_weight: float = 0.6,
        graph_weight: float = 0.4,
        max_nodes: int = 15,
        max_depth: int = 3
    ) -> Dict:
        """
        テキスト類似度とグラフ構造を組み合わせたハイブリッド検索
        
        Args:
            query: 検索クエリ
            text_weight: テキスト類似度の重み
            graph_weight: グラフ構造の重み
            max_nodes: 取得する最大ノード数
            max_depth: 探索する最大の深さ
            
        Returns:
            検索結果
        """
        # クエリの埋め込み
        query_embedding = self.compute_embedding(query)
        
        # Neo4jでのハイブリッド検索クエリ
        cypher_query = """
        // ステップ1: テキスト類似度の計算
        MATCH (n:Concept)
        WHERE n.embedding IS NOT NULL
        WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS text_similarity
        
        // ステップ2: グラフ中心性の計算
        CALL gds.pageRank.stream('esg_graph')
        YIELD nodeId, score AS centrality
        WHERE gds.util.asNode(nodeId) = n
        
        // ステップ3: ハイブリッドスコアの計算
        WITH n, text_similarity, centrality,
             $text_weight * text_similarity + $graph_weight * centrality AS hybrid_score
        ORDER BY hybrid_score DESC
        LIMIT $max_initial_nodes
        
        // ステップ4: コミュニティ検出とパス探索
        CALL gds.louvain.stream('esg_graph')
        YIELD nodeId, communityId
        WHERE gds.util.asNode(nodeId) = n
        WITH n, hybrid_score, communityId
        
        // ステップ5: サブグラフの抽出
        MATCH path = (n)-[r:ESG_RELATION*1..%d]-(m)
        WHERE ALL(rel IN r WHERE rel.confidence IS NULL OR rel.confidence >= 0.5)
        WITH COLLECT(path) AS paths, n, hybrid_score, communityId
        
        // ステップ6: 結果の整形
        UNWIND paths AS p
        WITH DISTINCT nodes(p) AS nodes, relationships(p) AS rels,
             n AS seed_node, hybrid_score, communityId
        RETURN 
            {
                seed_node: seed_node.name,
                score: hybrid_score,
                community: communityId,
                nodes: [n IN nodes | {
                    id: id(n),
                    name: n.name,
                    category: n.category,
                    properties: properties(n)
                }],
                relationships: [r IN rels | {
                    source: startNode(r).name,
                    type: r.type,
                    target: endNode(r).name,
                    properties: properties(r)
                }]
            } AS result
        LIMIT %d
        """ % (max_depth, max_nodes)
        
        params = {
            "query_embedding": query_embedding.tolist(),
            "text_weight": text_weight,
            "graph_weight": graph_weight,
            "max_initial_nodes": max_nodes
        }
        
        with self.neo4j.driver.session() as session:
            results = list(session.run(cypher_query, params))
            
            if not results:
                return {
                    "results": [],
                    "statistics": {
                        "total_results": 0,
                        "query_coverage": 0.0
                    }
                }
            
            # 結果の後処理とスコアリング
            processed_results = []
            for record in results:
                result = record["result"]
                # コンテキストの関連性スコアを計算
                context_score = self._compute_context_relevance(
                    query=query,
                    nodes=result["nodes"],
                    relationships=result["relationships"]
                )
                result["context_score"] = context_score
                processed_results.append(result)
            
            # クエリカバレッジの計算
            query_tokens = set(query.split())
            covered_tokens = set()
            for result in processed_results:
                for node in result["nodes"]:
                    node_tokens = set(node["name"].split())
                    covered_tokens.update(node_tokens)
            
            query_coverage = len(covered_tokens.intersection(query_tokens)) / len(query_tokens)
            
            return {
                "results": processed_results,
                "statistics": {
                    "total_results": len(processed_results),
                    "query_coverage": query_coverage
                }
            }
    
    def _compute_context_relevance(
        self,
        query: str,
        nodes: List[Dict],
        relationships: List[Dict]
    ) -> float:
        """コンテキストの関連性スコアを計算"""
        # クエリの埋め込み
        query_embedding = self.compute_embedding(query)
        
        # ノード名の埋め込みを計算
        node_texts = [node["name"] for node in nodes]
        node_embeddings = self.embedding_model.encode(
            node_texts,
            convert_to_tensor=True,
            device=self.device
        )
        
        # 関係の説明文の埋め込みを計算
        rel_texts = []
        for rel in relationships:
            description = rel.get("properties", {}).get("description", "")
            if description:
                rel_texts.append(description)
        
        if rel_texts:
            rel_embeddings = self.embedding_model.encode(
                rel_texts,
                convert_to_tensor=True,
                device=self.device
            )
            # ノードと関係の埋め込みを結合
            all_embeddings = torch.cat([node_embeddings, rel_embeddings], dim=0)
        else:
            all_embeddings = node_embeddings
        
        # コサイン類似度の計算
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            all_embeddings,
            dim=1
        )
        
        # 最大類似度と平均類似度の加重平均
        max_sim = similarities.max().item()
        mean_sim = similarities.mean().item()
        
        return 0.7 * max_sim + 0.3 * mean_sim
    
    def get_concept_hierarchy(
        self,
        concept: str,
        max_depth: int = 3
    ) -> Dict:
        """
        概念の階層構造を取得
        
        Args:
            concept: 対象の概念
            max_depth: 最大深さ
            
        Returns:
            階層構造の情報
        """
        cypher_query = """
        // 上位概念の取得
        MATCH path1 = (c:Concept {name: $concept})-[r1:ESG_RELATION*1..%d]->(parent)
        WHERE ALL(rel IN r1 WHERE rel.type = 'is_a')
        WITH COLLECT(path1) AS up_paths
        
        // 下位概念の取得
        MATCH path2 = (c:Concept {name: $concept})<-[r2:ESG_RELATION*1..%d]-(child)
        WHERE ALL(rel IN r2 WHERE rel.type = 'is_a')
        WITH up_paths, COLLECT(path2) AS down_paths
        
        // 結果の整形
        RETURN {
            up_paths: [path IN up_paths | [n IN nodes(path) | n.name]],
            down_paths: [path IN down_paths | [n IN nodes(path) | n.name]]
        } AS hierarchy
        """ % (max_depth, max_depth)
        
        with self.neo4j.driver.session() as session:
            result = session.run(cypher_query, concept=concept)
            hierarchy = result.single()["hierarchy"]
            
            return {
                "concept": concept,
                "parents": hierarchy["up_paths"],
                "children": hierarchy["down_paths"]
            } 
"""
Neo4jデータベースとの連携を管理するモジュール
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import json
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import logging
from dotenv import load_dotenv
import os

class Neo4jManager:
    """Neo4jデータベースの操作を管理するクラス"""
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """
        初期化
        
        Args:
            uri: Neo4jデータベースのURI
            user: ユーザー名
            password: パスワード
        """
        # 環境変数から認証情報を取得
        load_dotenv()
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        
        if not all([self.uri, self.user, self.password]):
            raise ValueError("Database credentials not properly configured")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        # インデックスの作成
        self._ensure_indexes()
    
    def _ensure_indexes(self) -> None:
        """必要なインデックスを作成"""
        indexes = [
            "CREATE INDEX esg_concept IF NOT EXISTS FOR (n:Concept) ON (n.name)",
            "CREATE INDEX esg_instance IF NOT EXISTS FOR (n:Instance) ON (n.name)",
            "CREATE INDEX esg_relation IF NOT EXISTS FOR ()-[r:ESG_RELATION]-() ON (r.type)",
        ]
        
        with self.driver.session() as session:
            for index_query in indexes:
                session.run(index_query)
    
    def add_concept(self, name: str, category: str, attributes: Optional[Dict] = None) -> None:
        """
        概念ノードを追加
        
        Args:
            name: 概念名
            category: カテゴリ（Environment/Social/Governance）
            attributes: 追加の属性
        """
        query = """
        MERGE (c:Concept {name: $name})
        SET c.category = $category
        """
        
        if attributes:
            query += " SET c += $attributes"
        
        with self.driver.session() as session:
            session.run(query, name=name, category=category, attributes=attributes or {})
    
    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Optional[Dict] = None
    ) -> None:
        """
        関係を追加
        
        Args:
            source: 関係の起点となるノード名
            target: 関係の終点となるノード名
            relation_type: 関係の種類
            properties: 関係の属性
        """
        query = """
        MATCH (s {name: $source})
        MATCH (t {name: $target})
        MERGE (s)-[r:ESG_RELATION {type: $relation_type}]->(t)
        """
        
        if properties:
            query += " SET r += $properties"
        
        with self.driver.session() as session:
            session.run(
                query,
                source=source,
                target=target,
                relation_type=relation_type,
                properties=properties or {}
            )
    
    def import_ontology(self, ontology_path: Path) -> None:
        """
        オントロジーファイルをインポート
        
        Args:
            ontology_path: オントロジーJSONファイルのパス
        """
        with open(ontology_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # ノードの作成
        for node in data["nodes"]:
            # カテゴリの判定
            category = "Other"
            if node in data["concepts"].get("ESG", {}).get("Environment", {}):
                category = "Environment"
            elif node in data["concepts"].get("ESG", {}).get("Social", {}):
                category = "Social"
            elif node in data["concepts"].get("ESG", {}).get("Governance", {}):
                category = "Governance"
            
            self.add_concept(node, category)
        
        # 関係の作成
        for edge in data["edges"]:
            self.add_relation(
                edge["source"],
                edge["target"],
                edge["relation"],
                edge.get("properties", {})
            )
    
    def get_related_concepts(
        self,
        concept: str,
        relation_type: Optional[str] = None,
        max_depth: int = 2
    ) -> List[Dict]:
        """
        指定した概念に関連する概念を取得
        
        Args:
            concept: 起点となる概念名
            relation_type: 関係の種類（指定しない場合は全ての関係）
            max_depth: 探索する最大の深さ
            
        Returns:
            関連する概念のリスト
        """
        query = """
        MATCH path = (s {name: $concept})-[r:ESG_RELATION*1..%d]->(t)
        """ % max_depth
        
        if relation_type:
            query += " WHERE ALL(rel IN r WHERE rel.type = $relation_type)"
        
        query += """
        RETURN t.name as target,
               [rel IN r | rel.type] as relations,
               length(path) as depth
        ORDER BY depth
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                concept=concept,
                relation_type=relation_type
            )
            
            return [
                {
                    "target": record["target"],
                    "relations": record["relations"],
                    "depth": record["depth"]
                }
                for record in result
            ]
    
    def search_concepts(
        self,
        keyword: str,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        キーワードで概念を検索
        
        Args:
            keyword: 検索キーワード
            category: カテゴリでフィルタ（任意）
            
        Returns:
            マッチした概念のリスト
        """
        query = """
        MATCH (c:Concept)
        WHERE c.name CONTAINS $keyword
        """
        
        if category:
            query += " AND c.category = $category"
        
        query += " RETURN c.name as name, c.category as category"
        
        with self.driver.session() as session:
            result = session.run(query, keyword=keyword, category=category)
            return [dict(record) for record in result]
    
    def close(self) -> None:
        """接続を閉じる"""
        self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 
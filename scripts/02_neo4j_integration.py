#!/usr/bin/env python3
"""
ESGオントロジーのNeo4jへの統合サンプルスクリプト
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.knowledge_graph.neo4j_manager import Neo4jManager
from src.knowledge_graph.ontology import ESGOntology

def main():
    # 環境変数の読み込み
    load_dotenv()
    
    # デバッグ出力
    print("Current working directory:", os.getcwd())
    print("Project root:", project_root)
    print(".env file exists:", Path(".env").exists())
    print("NEO4J_PASSWORD:", os.getenv("NEO4J_PASSWORD"))
    
    # Neo4j接続情報の確認
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORDが設定されていません。.envファイルを確認してください。")

    # Neo4jマネージャーの初期化
    print("Neo4jに接続中...")
    neo4j = Neo4jManager(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )

    # オントロジーの読み込み
    ontology_path = project_root / "data" / "processed" / "esg_ontology.json"
    print(f"\nオントロジーを読み込み中: {ontology_path}")
    ontology = ESGOntology.load(ontology_path)

    # オントロジーのインポート
    print("\nNeo4jにオントロジーをインポート中...")
    neo4j.import_ontology(ontology_path)

    # 概念間の関係探索
    print("\n気候変動に関連する概念を検索中...")
    climate_related = neo4j.get_related_concepts("気候変動", max_depth=2)
    print("\n気候変動に関連する概念:")
    for concept in climate_related:
        print(f"- {concept['target']} (関係: {' -> '.join(concept['relations'])}, 深さ: {concept['depth']})")

    # 特定のカテゴリ内での検索
    print("\n環境カテゴリ内のエネルギー関連概念を検索中...")
    env_concepts = neo4j.search_concepts(keyword="エネルギー", category="Environment")
    print("\n環境カテゴリ内のエネルギー関連概念:")
    for concept in env_concepts:
        print(f"- {concept['name']}")

    # 新しい関係の追加
    print("\n新しい関係を追加中...")
    neo4j.add_relation(
        source="再生可能エネルギー",
        target="カーボンニュートラル",
        relation_type="contributes_to",
        properties={"confidence": 0.9}
    )

    # 追加した関係の確認
    print("\n再生可能エネルギーが貢献する概念を確認中...")
    related = neo4j.get_related_concepts("再生可能エネルギー", relation_type="contributes_to")
    print("\n再生可能エネルギーが貢献する概念:")
    for concept in related:
        print(f"- {concept['target']}")

    # 可視化用のCypherクエリ
    visualization_query = """
    MATCH path = (c1:Concept)-[r:ESG_RELATION*1..2]->(c2:Concept)
    WHERE c1.category = 'Environment'
    RETURN path
    LIMIT 50
    """
    print("\nNeo4jブラウザで以下のクエリを実行することで関係を可視化できます:")
    print(visualization_query)

    # リソースの解放
    neo4j.close()
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main() 
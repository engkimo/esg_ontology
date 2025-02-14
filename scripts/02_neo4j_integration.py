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

def create_test_data(neo4j):
    """テストデータの作成"""
    # ESGレポートフレームワークの構造
    neo4j.add_node("ESG_Report_Framework", "Framework")
    neo4j.add_node("ESG_Category", "Category")
    neo4j.add_node("ESG_Metric", "Metric")
    neo4j.add_node("ESG_ComputingModel", "Model")
    neo4j.add_node("Dataset", "Data")
    neo4j.add_node("Indicator", "Metric")
    neo4j.add_node("Datasource", "Data")

    # 環境カテゴリの例
    neo4j.add_node("Environmental", "Category")
    neo4j.add_node("Climate_Change", "Category")
    neo4j.add_node("GHG_Emissions", "Metric")
    neo4j.add_node("Energy_Usage", "Metric")
    
    # 社会カテゴリの例
    neo4j.add_node("Social", "Category")
    neo4j.add_node("Human_Rights", "Category")
    neo4j.add_node("Labor_Practices", "Metric")
    
    # ガバナンスカテゴリの例
    neo4j.add_node("Governance", "Category")
    neo4j.add_node("Board_Structure", "Category")
    neo4j.add_node("Executive_Compensation", "Metric")

    # 関係性の定義
    relationships = [
        ("ESG_Report_Framework", "DividedInto", "ESG_Category"),
        ("ESG_Category", "Subcategory", "ESG_Category"),
        ("ESG_Category", "ESG_Category", "ESG_Metric"),
        ("ESG_Metric", "ObtainedFrom", "Dataset"),
        ("ESG_Metric", "DependentVariable", "ESG_ComputingModel"),
        ("ESG_ComputingModel", "DependentVariable", "Indicator"),
        ("Dataset", "DataSource", "Datasource"),
        ("Indicator", "Dataset", "Dataset"),
        
        # 環境カテゴリの関係
        ("Environmental", "Subcategory", "Climate_Change"),
        ("Climate_Change", "ESG_Category", "GHG_Emissions"),
        ("Climate_Change", "ESG_Category", "Energy_Usage"),
        
        # 社会カテゴリの関係
        ("Social", "Subcategory", "Human_Rights"),
        ("Human_Rights", "ESG_Category", "Labor_Practices"),
        
        # ガバナンスカテゴリの関係
        ("Governance", "Subcategory", "Board_Structure"),
        ("Board_Structure", "ESG_Category", "Executive_Compensation")
    ]

    # 関係性の追加
    for source, rel_type, target in relationships:
        neo4j.add_relation(
            source=source,
            target=target,
            relation_type=rel_type,
            properties={"confidence": 0.9}
        )

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

    # データベースのクリア
    print("\nデータベースをクリア中...")
    neo4j.clear_database()

    # テストデータの作成
    print("\nテストデータを作成中...")
    create_test_data(neo4j)

    # 可視化用のCypherクエリ
    visualization_query = """
    MATCH path = (n)-[r]->(m)
    RETURN path
    """
    print("\nNeo4jブラウザで以下のクエリを実行することで関係を可視化できます:")
    print(visualization_query)

    # リソースの解放
    neo4j.close()
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main() 
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
    # ESGの基本カテゴリ
    neo4j.add_node("ESG", "Framework")
    neo4j.add_node("Environmental", "Category")
    neo4j.add_node("Social", "Category")
    neo4j.add_node("Governance", "Category")

    # 環境カテゴリの詳細化
    env_categories = [
        "Climate_Change",
        "Resource_Management",
        "Biodiversity",
        "Pollution_Prevention",
        "Environmental_Management"
    ]
    env_metrics = {
        "Climate_Change": [
            "GHG_Emissions",
            "Energy_Efficiency",
            "Renewable_Energy",
            "Carbon_Pricing",
            "Climate_Risk_Management"
        ],
        "Resource_Management": [
            "Water_Usage",
            "Waste_Management",
            "Circular_Economy",
            "Resource_Efficiency"
        ],
        "Biodiversity": [
            "Ecosystem_Protection",
            "Species_Conservation",
            "Natural_Capital",
            "Land_Use"
        ]
    }

    # 社会カテゴリの詳細化
    social_categories = [
        "Human_Rights",
        "Labor_Practices",
        "Community_Relations",
        "Product_Responsibility",
        "Supply_Chain_Management"
    ]
    social_metrics = {
        "Human_Rights": [
            "Human_Rights_Assessment",
            "Indigenous_Rights",
            "Child_Labor_Prevention",
            "Forced_Labor_Prevention"
        ],
        "Labor_Practices": [
            "Occupational_Health_Safety",
            "Employee_Development",
            "Diversity_Inclusion",
            "Fair_Compensation"
        ],
        "Supply_Chain_Management": [
            "Supplier_Assessment",
            "Supply_Chain_Transparency",
            "Responsible_Sourcing",
            "Supplier_Engagement"
        ]
    }

    # ガバナンスカテゴリの詳細化
    gov_categories = [
        "Board_Structure",
        "Risk_Management",
        "Business_Ethics",
        "Compliance",
        "Stakeholder_Engagement"
    ]
    gov_metrics = {
        "Board_Structure": [
            "Board_Independence",
            "Board_Diversity",
            "Board_Effectiveness",
            "Executive_Compensation"
        ],
        "Risk_Management": [
            "Risk_Assessment",
            "Internal_Control",
            "Crisis_Management",
            "ESG_Risk_Integration"
        ],
        "Business_Ethics": [
            "Anti_Corruption",
            "Whistleblower_Protection",
            "Ethical_Guidelines",
            "Tax_Transparency"
        ]
    }

    # 関係性の種類を定義
    relationships = [
        # 基本構造
        ("Environmental", "Category_Of", "ESG"),
        ("Social", "Category_Of", "ESG"),
        ("Governance", "Category_Of", "ESG"),

        # 環境カテゴリの関係
        *[(cat, "Subcategory_Of", "Environmental") for cat in env_categories],
        *[(metric, "Metric_Of", cat) for cat, metrics in env_metrics.items() for metric in metrics],

        # 社会カテゴリの関係
        *[(cat, "Subcategory_Of", "Social") for cat in social_categories],
        *[(metric, "Metric_Of", cat) for cat, metrics in social_metrics.items() for metric in metrics],

        # ガバナンスカテゴリの関係
        *[(cat, "Subcategory_Of", "Governance") for cat in gov_categories],
        *[(metric, "Metric_Of", cat) for cat, metrics in gov_metrics.items() for metric in metrics],

        # 相互関係
        ("Climate_Change", "Impacts", "Community_Relations"),
        ("Supply_Chain_Management", "Influences", "GHG_Emissions"),
        ("Risk_Management", "Monitors", "Climate_Risk_Management"),
        ("Board_Structure", "Oversees", "ESG_Risk_Integration"),
        ("Stakeholder_Engagement", "Enhances", "Community_Relations"),
        ("Business_Ethics", "Strengthens", "Supply_Chain_Management")
    ]

    # ノードの追加
    print("\nノードを追加中...")
    # 環境カテゴリ
    for cat in env_categories:
        neo4j.add_node(cat, "Environmental")
        for metric in env_metrics.get(cat, []):
            neo4j.add_node(metric, "Metric")

    # 社会カテゴリ
    for cat in social_categories:
        neo4j.add_node(cat, "Social")
        for metric in social_metrics.get(cat, []):
            neo4j.add_node(metric, "Metric")

    # ガバナンスカテゴリ
    for cat in gov_categories:
        neo4j.add_node(cat, "Governance")
        for metric in gov_metrics.get(cat, []):
            neo4j.add_node(metric, "Metric")

    # 関係性の追加
    print("\n関係性を追加中...")
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
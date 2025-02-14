#!/usr/bin/env python3
"""
ESG知識グラフの補完サンプルスクリプト
"""

import sys
from pathlib import Path
import torch
from dotenv import load_dotenv
import os
import networkx as nx

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.knowledge_completion import ESGKnowledgeCompletion
from src.knowledge_graph.neo4j_manager import Neo4jManager
from src.utils.device import get_device

def main():
    # 環境変数の読み込み
    load_dotenv()
    
    # デバイスの確認
    device = get_device()
    print(f"Using device: {device}")

    # Neo4j接続情報の確認
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORDが設定されていません。.envファイルを確認してください。")

    # 知識補完モデルの初期化
    print("\n知識補完モデルを初期化中...")
    completion_model = ESGKnowledgeCompletion(
        llm_model_name="rinna/japanese-gpt-neox-small",
        device=device
    )

    # Neo4jからグラフ構造を取得
    print("\nNeo4jからグラフ構造を取得中...")
    neo4j = Neo4jManager(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )

    # サンプルグラフの作成
    print("\nサンプルデータでGNNを学習中...")
    G = nx.DiGraph()
    G.add_nodes_from(['気候変動', '温室効果ガス', '再生可能エネルギー', 'カーボンニュートラル'])
    G.add_edges_from([
        ('温室効果ガス', '気候変動'),
        ('再生可能エネルギー', 'カーボンニュートラル'),
        ('気候変動', 'カーボンニュートラル')
    ])

    # GNNの学習
    completion_model.train(
        graph=G,
        num_epochs=100,
        batch_size=32
    )

    # リンク予測
    print("\nリンク予測を実行中...")
    predictions = completion_model.predict_links(
        graph=G,
        source_node='気候変動',
        top_k=5
    )

    print("\n予測された新しいリンク（上位5件）:")
    for target, score in predictions:
        print(f"{target}: {score:.4f}")

    # LLMによる関係推論
    print("\nLLMによる関係推論を実行中...")
    concepts = [
        "気候変動",
        "サプライチェーンマネジメント",
        "コーポレートガバナンス"
    ]

    for concept in concepts:
        print(f"\n{concept}に関する推論:")
        relations = completion_model.infer_relations_with_llm(
            source=concept,
            context=f"{concept}は重要なESG課題の一つです。"
        )
        
        for relation in relations:
            print(f"\n関連概念: {relation['target']}")
            print(f"関係: {relation['relation']}")
            print(f"説明: {relation['description']}")

            # 推論された関係をNeo4jに追加
            neo4j.add_relation(
                source=concept,
                target=relation["target"],
                relation_type=relation["relation"],
                properties={"description": relation["description"]}
            )

    # リソースの解放
    neo4j.close()
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main() 
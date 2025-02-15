#!/usr/bin/env python3
"""
ESG知識グラフの補完サンプルスクリプト
"""

import sys
from pathlib import Path
import torch
from dotenv import load_dotenv
import os

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

    # エッジ情報の準備（サンプル）
    print("\nサンプルデータでGNNを学習中...")
    edge_index = torch.tensor([
        [0, 1, 1, 2],  # source nodes
        [1, 2, 3, 3]   # target nodes
    ], device=device)
    num_nodes = 4

    # GNNの学習
    completion_model.train_gnn(
        edge_index=edge_index,
        num_nodes=num_nodes,
        epochs=100,
        batch_size=32
    )

    # リンク予測
    print("\nリンク予測を実行中...")
    prob, pairs = completion_model.predict_links_with_gnn(
        edge_index=edge_index,
        num_nodes=num_nodes
    )

    print("\n予測された新しいリンク（上位5件）:")
    for i in range(min(5, len(prob))):
        print(f"Node {pairs[0,i]} -> Node {pairs[1,i]}: {prob[i]:.4f}")

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
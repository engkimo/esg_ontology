#!/usr/bin/env python3
"""
ESGオントロジーシステムの評価スクリプト
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from src.rag.graph_rag import ESGGraphRAG

def load_test_data(file_path: str = "data/evaluation/test_cases.json") -> dict:
    """テストデータの読み込み"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_relation_inference(rag: ESGGraphRAG, test_cases: list) -> dict:
    """関係推論の評価"""
    results = {
        "correct": 0,
        "total": len(test_cases),
        "details": []
    }
    
    for case in test_cases:
        # サブグラフの検索
        subgraph = rag.search_relevant_subgraph(
            query=case["query"],
            max_nodes=10,
            max_depth=2
        )
        
        # 予測された関係と正解の関係を比較
        predicted_rels = set(
            f"{rel['source']}-{rel['type']}-{rel['target']}"
            for rel in subgraph["relationships"]
        )
        
        gold_rels = set(
            f"{rel['source']}-{rel['type']}-{rel['target']}"
            for rel in case["gold_relations"]
        )
        
        # 正解数をカウント
        correct = len(predicted_rels.intersection(gold_rels))
        if correct > 0:
            results["correct"] += 1
        
        # 詳細を記録
        results["details"].append({
            "query": case["query"],
            "predicted": list(predicted_rels),
            "gold": list(gold_rels),
            "correct": correct
        })
    
    # 精度を計算
    results["accuracy"] = results["correct"] / results["total"]
    return results

def evaluate_rag_queries(rag: ESGGraphRAG, test_cases: list) -> dict:
    """RAGクエリの評価"""
    results = {
        "node_coverage": [],
        "relation_coverage": [],
        "details": []
    }
    
    for case in test_cases:
        # サブグラフの検索
        subgraph = rag.search_relevant_subgraph(
            query=case["query"],
            max_nodes=15,
            max_depth=3
        )
        
        # 回答の生成
        response = rag.generate_response(
            query=case["query"],
            subgraph=subgraph
        )
        
        # カバレッジの計算
        found_nodes = set(node["name"] for node in subgraph["nodes"])
        gold_nodes = set(case["context_concepts"])
        node_coverage = len(found_nodes.intersection(gold_nodes)) / len(gold_nodes)
        
        found_rels = set(
            f"{rel['source']}-{rel['type']}-{rel['target']}"
            for rel in subgraph["relationships"]
        )
        gold_rels = set(
            f"{rel['source']}-{rel['type']}-{rel['target']}"
            for rel in case["context_relations"]
        )
        rel_coverage = len(found_rels.intersection(gold_rels)) / len(gold_rels)
        
        results["node_coverage"].append(node_coverage)
        results["relation_coverage"].append(rel_coverage)
        
        # 詳細を記録
        results["details"].append({
            "query": case["query"],
            "response": response,
            "reference": case["reference_answer"],
            "node_coverage": node_coverage,
            "relation_coverage": rel_coverage
        })
    
    # 平均カバレッジを計算
    results["avg_node_coverage"] = np.mean(results["node_coverage"])
    results["avg_relation_coverage"] = np.mean(results["relation_coverage"])
    return results

def visualize_results(results: dict, output_dir: str):
    """評価結果の可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 関係推論の結果をプロット
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=["Accuracy"],
        y=[results["relation_inference"]["accuracy"]]
    )
    plt.title("Relation Inference Accuracy")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, "relation_inference.png"))
    plt.close()
    
    # RAGクエリの結果をプロット
    coverage_data = pd.DataFrame({
        "Type": ["Node Coverage", "Relation Coverage"],
        "Value": [
            results["rag_queries"]["avg_node_coverage"],
            results["rag_queries"]["avg_relation_coverage"]
        ]
    })
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=coverage_data, x="Type", y="Value")
    plt.title("Average Coverage Metrics")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, "coverage_metrics.png"))
    plt.close()

def main():
    """メイン関数"""
    # 環境変数の読み込み
    load_dotenv()
    
    # 出力ディレクトリの設定
    output_dir = "data/evaluation/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # GraphRAGシステムの初期化
    rag = ESGGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )
    
    try:
        # テストデータの読み込み
        print("テストデータを読み込んでいます...")
        test_data = load_test_data()
        
        # 評価の実行
        print("関係推論の評価を実行中...")
        relation_results = evaluate_relation_inference(
            rag=rag,
            test_cases=test_data["relation_inference_cases"]
        )
        
        print("RAGクエリの評価を実行中...")
        rag_results = evaluate_rag_queries(
            rag=rag,
            test_cases=test_data["rag_cases"]
        )
        
        # 結果の集計
        results = {
            "relation_inference": relation_results,
            "rag_queries": rag_results
        }
        
        # 結果の保存
        print("結果を保存しています...")
        with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 結果の可視化
        print("結果を可視化しています...")
        visualize_results(results, output_dir)
        
        print("評価が完了しました。")
        print(f"結果は {output_dir} に保存されました。")
        
    finally:
        # リソースの解放
        rag.close()

if __name__ == "__main__":
    main() 
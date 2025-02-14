#!/usr/bin/env python3
"""
ESG GraphRAGシステムのサンプルスクリプト
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import networkx as nx
import torch
import warnings
import matplotlib
import json

# 警告メッセージの抑制
warnings.filterwarnings("ignore", category=UserWarning)

# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'Hiragino Sans'  # macOSの場合
# matplotlib.rcParams['font.family'] = 'IPAGothic'    # Linuxの場合
# matplotlib.rcParams['font.family'] = 'MS Gothic'    # Windowsの場合

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.rag.graph_rag import ESGGraphRAG
from src.utils.device import get_device

def visualize_subgraph(subgraph: dict, title: str):
    """サブグラフの可視化"""
    G = nx.DiGraph()
    
    # ノードの追加
    for node in subgraph["nodes"]:
        G.add_node(node["name"], category=node["category"])
    
    # エッジの追加
    for rel in subgraph["relationships"]:
        G.add_edge(rel["source"], rel["target"], type=rel["type"])
    
    # 描画設定
    plt.figure(figsize=(20, 15))
    
    # フォント設定
    plt.rcParams['font.family'] = 'Hiragino Sans'  # macOSの日本語フォント
    
    # レイアウトの設定（階層的レイアウトを使用）
    pos = nx.spring_layout(G, k=3, iterations=50)  # ノード間の距離を広げる
    
    # カテゴリ別の色とスタイル設定
    colors = {
        "Environmental": {"color": "#2ecc71", "alpha": 0.7},  # 緑
        "Social": {"color": "#3498db", "alpha": 0.7},        # 青
        "Governance": {"color": "#e74c3c", "alpha": 0.7},    # 赤
        "Metric": {"color": "#f1c40f", "alpha": 0.7},        # 黄
        "Framework": {"color": "#9b59b6", "alpha": 0.7},     # 紫
        "Category": {"color": "#1abc9c", "alpha": 0.7},      # ターコイズ
        "Other": {"color": "#95a5a6", "alpha": 0.7}          # グレー
    }
    
    # カテゴリ別のノード描画
    for category, style in colors.items():
        nodes = [n for n, attr in G.nodes(data=True) if attr.get("category") == category]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=style["color"],
                node_size=4000,
                alpha=style["alpha"],
                edgecolors='white',
                linewidths=2
            )
    
    # エッジの描画（関係タイプごとに色分け）
    edge_colors = {
        "Category_Of": "#2c3e50",      # 濃紺
        "Subcategory_Of": "#8e44ad",   # 紫
        "Metric_Of": "#d35400",        # オレンジ
        "Impacts": "#27ae60",          # 緑
        "Influences": "#c0392b",       # 赤
        "Monitors": "#16a085",         # ターコイズ
        "Oversees": "#2980b9",         # 青
        "Enhances": "#f39c12",         # 黄
        "Strengthens": "#7f8c8d"       # グレー
    }
    
    # エッジをタイプごとに描画
    for edge_type, color in edge_colors.items():
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("type") == edge_type]
        if edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                edge_color=color,
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                width=2,
                alpha=0.6
            )
    
    # ノードラベルの描画
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_family='Hiragino Sans',
        font_weight='bold'
    )
    
    # エッジラベルの描画
    edge_labels = nx.get_edge_attributes(G, "type")
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=10,
        font_family='Hiragino Sans',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )
    
    # タイトルとレジェンドの設定
    plt.title(title, fontsize=16, pad=20, fontfamily='Hiragino Sans')
    
    # カテゴリとエッジタイプのレジェンド
    node_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=style["color"],
                  markersize=15,
                  alpha=style["alpha"],
                  label=f"Node: {category}")
        for category, style in colors.items()
        if any(attr.get("category") == category for _, attr in G.nodes(data=True))
    ]
    
    edge_legend_elements = [
        plt.Line2D([0], [0], color=color, alpha=0.6,
                  label=f"Edge: {edge_type}")
        for edge_type, color in edge_colors.items()
        if any(d.get("type") == edge_type for _, _, d in G.edges(data=True))
    ]
    
    plt.legend(
        handles=node_legend_elements + edge_legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=10
    )
    
    plt.axis("off")
    plt.tight_layout()
    
    # 画像として保存
    output_dir = project_root / "data" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / f"{title.replace(' ', '_').lower()}.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()

def evaluate_response(question: str, answer: dict, subgraph: dict) -> dict:
    """回答の品質評価"""
    # 使用された概念数
    used_concepts = len(subgraph["nodes"])
    
    # カテゴリのカバレッジ
    categories = set(node["category"] for node in subgraph["nodes"])
    category_coverage = len(categories)
    
    # 関係の多様性
    relation_types = set(rel["type"] for rel in subgraph["relationships"])
    relation_diversity = len(relation_types)
    
    # 回答の充実度
    initiatives_count = len(answer.get("key_initiatives", []))
    challenges_count = len(answer.get("challenges", []))
    metrics_count = len(answer.get("metrics", []))
    references_count = len(answer.get("references", []))
    
    return {
        "概念数": used_concepts,
        "カテゴリカバレッジ": category_coverage,
        "関係性の多様性": relation_diversity,
        "施策数": initiatives_count,
        "課題数": challenges_count,
        "指標数": metrics_count,
        "参照概念数": references_count,
        "使用カテゴリ": list(categories),
        "使用関係タイプ": list(relation_types)
    }

def print_structured_response(response: dict):
    """構造化された回答を表示"""
    print("\n【概要】")
    print(response["overview"])
    
    print("\n【主要な施策】")
    for i, initiative in enumerate(response["key_initiatives"], 1):
        print(f"\n{i}. {initiative['title']}")
        print(f"   説明: {initiative['description']}")
        print(f"   実施方法: {initiative['implementation']}")
    
    print("\n【課題と注意点】")
    for i, challenge in enumerate(response["challenges"], 1):
        print(f"{i}. {challenge}")
    
    print("\n【評価指標と目標】")
    for i, metric in enumerate(response["metrics"], 1):
        print(f"\n{i}. {metric['name']}")
        print(f"   目標: {metric['target']}")
        print(f"   期間: {metric['timeline']}")
    
    print("\n【まとめ】")
    print(f"要約: {response['conclusion']['summary']}")
    print(f"展望: {response['conclusion']['future_outlook']}")
    
    print("\n【参照概念】")
    for ref in response["references"]:
        print(f"- {ref['concept']} ({ref['category']})")
        print(f"  関連性: {ref['relevance']}")

def main():
    # 環境変数の読み込み
    load_dotenv()
    
    # 必要な環境変数の確認
    required_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OPENAI_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"{var}が設定されていません。.envファイルを確認してください。")

    # デバイスの確認
    device = get_device()
    print(f"Using device: {device}")

    # GraphRAGの初期化
    print("\nGraphRAGシステムを初期化中...")
    rag = ESGGraphRAG(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_model_name="sonoisa/sentence-bert-base-ja-mean-tokens",
        device=device,
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )

    # ノード埋め込みの更新
    print("\nノード埋め込みを更新中...")
    rag.update_node_embeddings(batch_size=4)

    # ESGの各側面に関する具体的な質問
    questions = [
        # 環境（Environment）に関する質問
        "企業がScope 3排出量を削減するために取るべき具体的な施策を教えてください。",
        
        # 社会（Social）に関する質問
        "サプライチェーン全体での人権デューデリジェンスをどのように実施すべきですか？",
        
        # ガバナンス（Governance）に関する質問
        "取締役会の実効性を高めるために重要な要素は何ですか？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n質問 {i}: {question}")
        print("-" * 80)
        
        # 関連するサブグラフの検索
        subgraph = rag.search_relevant_subgraph(
            query=question,
            max_nodes=15,
            max_depth=3
        )
        
        # 回答の生成
        response = rag.generate_response(
            query=question,
            subgraph=subgraph,
            temperature=0.3
        )
        
        # 構造化された回答の表示
        print_structured_response(response)
        
        # 回答の評価
        evaluation = evaluate_response(question, response, subgraph)
        print("\n【回答の評価結果】")
        for metric, value in evaluation.items():
            print(f"- {metric}: {value}")
        
        # サブグラフの可視化
        visualize_subgraph(subgraph, f"Question {i} Knowledge Graph")

    # 特定のトピックの詳細分析
    topics = ["気候変動対策", "サプライチェーンマネジメント"]
    
    for topic in topics:
        print(f"\n{topic}の詳細分析:")
        
        # サブグラフの取得と可視化
        subgraph = rag.search_relevant_subgraph(
            query=topic,
            max_nodes=20,
            max_depth=3
        )
        
        # 概念のカテゴリ分布
        categories = {}
        for node in subgraph["nodes"]:
            cat = node["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n概念のカテゴリ分布:")
        for cat, count in categories.items():
            print(f"- {cat}: {count}概念")
        
        # 関係性の分析
        relation_types = {}
        for rel in subgraph["relationships"]:
            rel_type = rel["type"]
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        
        print(f"\n関係性の種類:")
        for rel_type, count in relation_types.items():
            print(f"- {rel_type}: {count}件")
        
        # グラフの可視化
        visualize_subgraph(subgraph, f"{topic} Analysis")

    # リソースの解放
    rag.close()
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main() 
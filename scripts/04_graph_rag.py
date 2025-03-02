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

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.rag.graph_rag import ESGGraphRAG
from src.utils.device import get_device

def visualize_subgraph(subgraph: dict, title: str):
    """サブグラフの可視化"""
    # フォント設定
    plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'Yu Gothic', 'MS Gothic', 'IPAGothic']
    
    G = nx.DiGraph()
    
    # ノードの追加
    for node in subgraph["nodes"]:
        G.add_node(node["name"], category=node["category"])
    
    # エッジの追加
    for rel in subgraph["relationships"]:
        G.add_edge(rel["source"], rel["target"], type=rel["type"])
    
    if len(G.nodes) == 0:
        print(f"警告: {title}のサブグラフにノードが存在しません。")
        return
    
    # 描画
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)  # kパラメータを調整
    
    # カテゴリ別の色分け
    colors = {
        "Environment": "#90EE90",  # 薄緑
        "Social": "#ADD8E6",      # 薄青
        "Governance": "#FFB6C1",  # 薄紅
        "Other": "#D3D3D3"       # 薄灰
    }
    
    # ノードの描画
    for category in colors:
        nodes = [n for n, attr in G.nodes(data=True) if attr.get("category") == category]
        if nodes:  # ノードが存在する場合のみ描画
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors[category],
                                 node_size=3000, alpha=0.7)  # ノードサイズを大きく
    
    # エッジとラベルの描画
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20,
                          width=2.0)  # エッジを太く
    
    # ノードラベルの描画（日本語対応）
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_family='Hiragino Sans', font_size=12)
    
    # エッジラベルの描画（日本語対応）
    edge_labels = nx.get_edge_attributes(G, "type")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_family='Hiragino Sans', font_size=10)
    
    plt.title(f"Query: {title}", fontsize=15, fontfamily='Hiragino Sans')
    plt.axis("off")
    plt.tight_layout()
    
    # 画像として保存
    output_dir = project_root / "data" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{title.replace(' ', '_').lower()}.png",
                bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_response(question: str, answer: str, subgraph: dict) -> dict:
    """回答の品質評価"""
    # 使用された概念数
    used_concepts = len(subgraph["nodes"])
    
    # カテゴリのカバレッジ
    categories = set(node["category"] for node in subgraph["nodes"])
    category_coverage = len(categories)
    
    # 関係の多様性
    relation_types = set(rel["type"] for rel in subgraph["relationships"])
    relation_diversity = len(relation_types)
    
    # 回答の長さ（文字数）
    answer_length = len(answer)
    
    return {
        "概念数": used_concepts,
        "カテゴリカバレッジ": category_coverage,
        "関係性の多様性": relation_diversity,
        "回答の長さ": answer_length,
        "使用カテゴリ": list(categories),
        "使用関係タイプ": list(relation_types)
    }

def main():
    # 環境変数の読み込み
    load_dotenv()
    
    # Neo4j接続情報の確認
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORDが設定されていません。.envファイルを確認してください。")

    # デバイスの確認
    device = get_device()
    print(f"Using device: {device}")

    # GraphRAGの初期化
    print("\nGraphRAGシステムを初期化中...")
    rag = ESGGraphRAG(
        llm_model_name="rinna/japanese-gpt-neox-small",
        embedding_model_name="sonoisa/sentence-bert-base-ja-mean-tokens",
        device=device,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        use_mps=True  # MPSデバイスを使用
    )

    # ノード埋め込みの更新
    print("\nノード埋め込みを更新中...")
    rag.update_node_embeddings(batch_size=16)  # MacBookのメモリを考慮してバッチサイズを調整

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
            max_nodes=15,  # より広い文脈を取得
            max_depth=3    # より深い関係性を探索
        )
        
        # 回答の生成
        answer = rag.generate_response(
            query=question,
            subgraph=subgraph
        )
        
        print(f"\n回答:\n{answer}")
        
        # 参照された知識の表示
        print("\n参照された主な概念:")
        for node in subgraph["nodes"][:5]:  # 主要な5つの概念のみ表示
            print(f"- {node['name']} ({node['category']})")
        
        print("\n主な関係性:")
        for rel in subgraph["relationships"][:3]:  # 主要な3つの関係のみ表示
            print(f"- {rel['source']} → {rel['type']} → {rel['target']}")
        
        # 回答の評価
        evaluation = evaluate_response(question, answer, subgraph)
        print("\n回答の評価結果:")
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
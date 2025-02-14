"""
知識グラフ補完モジュールのテスト
"""

import pytest
import networkx as nx
from src.models.knowledge_completion import ESGKnowledgeCompletion

def test_llm_relation_inference():
    """LLMによる関係推論のテスト"""
    model = ESGKnowledgeCompletion()
    
    # 気候変動に関する関係推論
    relations = model.infer_relations_with_llm("気候変動")
    assert len(relations) > 0
    
    # 各関係が必要な情報を含んでいることを確認
    for relation in relations:
        assert "target" in relation
        assert "relation" in relation
        assert "description" in relation

def test_gnn_link_prediction():
    """GNNによるリンク予測のテスト"""
    # テスト用のグラフ作成
    G = nx.DiGraph()
    nodes = ["気候変動", "温室効果ガス", "再生可能エネルギー", "省エネルギー"]
    G.add_nodes_from(nodes)
    edges = [
        ("気候変動", "温室効果ガス"),
        ("温室効果ガス", "再生可能エネルギー"),
        ("再生可能エネルギー", "省エネルギー")
    ]
    G.add_edges_from(edges)
    
    model = ESGKnowledgeCompletion()
    
    # モデルの学習
    model.train(G, num_epochs=5)  # テスト用に少ないエポック数
    
    # リンク予測
    predictions = model.predict_links(G, "気候変動", top_k=2)
    assert len(predictions) == 2
    
    # 予測結果の形式を確認
    for node, score in predictions:
        assert isinstance(node, str)
        assert isinstance(score, float)
        assert 0 <= score <= 1

if __name__ == "__main__":
    pytest.main([__file__]) 
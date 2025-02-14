"""
ESGドメインのオントロジー構造を定義するモジュール
"""

from typing import Dict, List, Optional, Set, Tuple
import json
from pathlib import Path
import networkx as nx

class ESGOntology:
    """ESGドメインのオントロジーを管理するクラス"""
    
    def __init__(self):
        """初期化"""
        # グラフ構造の初期化
        self.graph = nx.DiGraph()
        
        # 基本カテゴリの定義
        self.categories = {
            "Environment": {
                "keywords": [
                    "環境", "気候変動", "カーボン", "エネルギー", "廃棄物",
                    "リサイクル", "生物多様性", "温室効果", "再生可能"
                ]
            },
            "Social": {
                "keywords": [
                    "社会", "人権", "労働", "安全", "健康", "教育",
                    "ダイバーシティ", "地域", "コミュニティ"
                ]
            },
            "Governance": {
                "keywords": [
                    "ガバナンス", "取締役", "監査", "コンプライアンス",
                    "リスク", "内部統制", "株主", "経営"
                ]
            }
        }
        
        # 基本カテゴリをグラフに追加
        for category in self.categories:
            self.graph.add_node(category, type="category")
        
        # 概念とインスタンスの初期化
        self.concepts = {}  # 概念の辞書
        self.instances = {}  # インスタンスの辞書
        self.relations = []  # 関係のリスト
    
    def _build_initial_graph(self) -> None:
        """基本概念からグラフ構造を構築"""
        def add_concepts(parent: str, concepts: Dict, relation: str = "is_a"):
            if isinstance(concepts, dict):
                for concept, subconcepts in concepts.items():
                    # 親ノードと子ノードを追加
                    self.graph.add_node(parent)
                    self.graph.add_node(concept)
                    # 関係を追加
                    self.graph.add_edge(concept, parent, relation=relation)
                    # 再帰的に下位概念を追加
                    add_concepts(concept, subconcepts)
            elif isinstance(concepts, set):
                for concept in concepts:
                    # 親ノードと子ノードを追加
                    self.graph.add_node(parent)
                    self.graph.add_node(concept)
                    # 関係を追加
                    self.graph.add_edge(concept, parent, relation=relation)
        
        # ルートノードを追加
        self.graph.add_node("ROOT")
        # 基本概念の構築
        add_concepts("ROOT", self.concepts)
    
    def add_concept(
        self,
        concept: str,
        parent: str,
        relation: str = "is_a"
    ) -> None:
        """
        新しい概念を追加
        
        Args:
            concept: 追加する概念
            parent: 親概念
            relation: 関係の種類
        """
        # 概念の追加
        if concept not in self.concepts:
            self.concepts[concept] = {
                "category": parent,
                "relations": []
            }
        
        # グラフに追加
        self.graph.add_node(concept, type="concept", category=parent)
        self.graph.add_edge(concept, parent, relation=relation)
    
    def add_instance(
        self,
        instance: str,
        concept: str,
        attributes: Optional[Dict] = None
    ) -> None:
        """
        概念のインスタンスを追加
        
        Args:
            instance: インスタンス名
            concept: 所属する概念
            attributes: インスタンスの属性
        """
        if not self.graph.has_node(concept):
            raise ValueError(f"Concept not found: {concept}")
        
        self.graph.add_edge(instance, concept, relation="instance_of")
        if attributes:
            self.graph.nodes[instance]["attributes"] = attributes
    
    def get_subconcepts(self, concept: str) -> Set[str]:
        """
        指定した概念の下位概念を取得
        
        Args:
            concept: 対象の概念
            
        Returns:
            下位概念の集合
        """
        if not self.graph.has_node(concept):
            return set()
        
        subconcepts = set()
        for node in self.graph.nodes():
            # 指定された概念に向かうエッジを持つノードを探す
            if self.graph.has_edge(node, concept):
                edge_data = self.graph.get_edge_data(node, concept)
                if edge_data.get("relation") == "is_a":
                    subconcepts.add(node)
        
        return subconcepts
    
    def get_instances(self, concept: str) -> Set[str]:
        """
        指定した概念のインスタンスを取得
        
        Args:
            concept: 対象の概念
            
        Returns:
            インスタンスの集合
        """
        if not self.graph.has_node(concept):
            return set()
        
        return {
            node for node in self.graph.predecessors(concept)
            if self.graph[node][concept]["relation"] == "instance_of"
        }
    
    def save(self, file_path: Path) -> None:
        """
        オントロジーをJSONファイルとして保存
        
        Args:
            file_path: 保存先のパス
        """
        def convert_to_json_serializable(obj):
            """オブジェクトをJSON変換可能な形式に変換"""
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            return obj

        data = {
            "nodes": list(self.graph.nodes()),
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "relation": d["relation"]
                }
                for u, v, d in self.graph.edges(data=True)
            ],
            "concepts": convert_to_json_serializable(self.concepts),
            "relations": self.relations
        }
        
        # ディレクトリが存在しない場合は作成
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, file_path: Path) -> "ESGOntology":
        """
        JSONファイルからオントロジーを読み込み
        
        Args:
            file_path: 読み込むファイルのパス
            
        Returns:
            ESGOntologyインスタンス
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        ontology = cls()
        ontology.concepts = data["concepts"]
        ontology.relations = data["relations"]
        
        # グラフの再構築
        ontology.graph = nx.DiGraph()
        for node in data["nodes"]:
            ontology.graph.add_node(node)
        
        for edge in data["edges"]:
            ontology.graph.add_edge(
                edge["source"],
                edge["target"],
                relation=edge["relation"]
            )
        
        return ontology

    def classify_concept(self, concept: str) -> Optional[str]:
        """
        概念をESGカテゴリに分類
        
        Args:
            concept: 概念名
            
        Returns:
            カテゴリ名（"Environment"/"Social"/"Governance"）
        """
        # スコアの初期化
        scores = {
            "Environment": 0,
            "Social": 0,
            "Governance": 0
        }
        
        # 各カテゴリのキーワードとのマッチングでスコアを計算
        for category, data in self.categories.items():
            for keyword in data["keywords"]:
                if keyword in concept:
                    scores[category] += 1
        
        # 最高スコアのカテゴリを返す
        max_score = max(scores.values())
        if max_score > 0:
            for category, score in scores.items():
                if score == max_score:
                    return category
        
        return None 

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
            source: 開始ノード
            target: 終了ノード
            relation_type: 関係タイプ
            properties: 関係のプロパティ
        """
        # 関係の追加
        relation = {
            "source": source,
            "target": target,
            "type": relation_type,
            "properties": properties or {}
        }
        self.relations.append(relation)
        
        # グラフに追加
        self.graph.add_edge(
            source,
            target,
            relation=relation_type,
            **properties or {}
        ) 
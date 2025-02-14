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
        # 基本概念の階層構造を定義
        self.concepts = {
            "ESG": {
                "Environment": {
                    "気候変動": {"温室効果ガス", "カーボンニュートラル"},
                    "資源効率": {"再生可能エネルギー", "廃棄物管理"},
                    "生物多様性": {"生態系保護", "自然資本"}
                },
                "Social": {
                    "人権": {"労働権", "児童労働防止"},
                    "労働安全": {"労働環境", "健康管理"},
                    "地域社会": {"コミュニティ貢献", "社会的包摂"}
                },
                "Governance": {
                    "企業統治": {"取締役会", "株主権利"},
                    "リスク管理": {"内部統制", "コンプライアンス"},
                    "情報開示": {"透明性", "ESG情報開示"}
                }
            }
        }
        
        # 関係性の定義
        self.relations = {
            "is_a": "上位概念-下位概念の関係",
            "part_of": "全体-部分の関係",
            "affects": "影響を与える関係",
            "measured_by": "指標による測定関係",
            "regulated_by": "規制・基準による管理関係"
        }
        
        # グラフ構造の初期化
        self.graph = nx.DiGraph()
        self._build_initial_graph()
    
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
        if relation not in self.relations:
            raise ValueError(f"Unknown relation type: {relation}")
        
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
        for category, data in self.concepts.items():
            for keyword in data["keywords"]:
                if keyword in concept:
                    scores[category] += 1
        
        # 最高スコアのカテゴリを返す
        max_score = max(scores.values())
        if max_score > 0:
            for category, score in scores.items():
                if score == max_score:
                    return category 
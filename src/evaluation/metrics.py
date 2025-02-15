"""
ESGオントロジーシステムの評価指標モジュール
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import networkx as nx
import re
from collections import Counter

class OntologyEvaluator:
    """オントロジー構築の評価"""
    
    def __init__(self, gold_standard_path: Optional[str] = None):
        """
        初期化
        
        Args:
            gold_standard_path: 正解データのパス（オプション）
        """
        self.gold_standard = None
        if gold_standard_path:
            self.load_gold_standard(gold_standard_path)
    
    def structural_metrics(self, ontology: nx.DiGraph) -> Dict[str, float]:
        """
        オントロジーの構造的メトリクスを計算
        
        Args:
            ontology: 評価対象のオントロジー
            
        Returns:
            構造的メトリクスの辞書
        """
        metrics = {
            "depth": nx.dag_longest_path_length(ontology),
            "breadth": max(len(list(ontology.predecessors(node))) for node in ontology.nodes()),
            "avg_branching_factor": np.mean([len(list(ontology.predecessors(node))) for node in ontology.nodes()]),
            "density": nx.density(ontology),
            "leaf_count": len([n for n in ontology.nodes() if ontology.out_degree(n) == 0]),
            "root_count": len([n for n in ontology.nodes() if ontology.in_degree(n) == 0])
        }
        return metrics
    
    def coverage_metrics(self, ontology: nx.DiGraph, domain_concepts: Set[str]) -> Dict[str, float]:
        """
        ドメインの網羅性を評価
        
        Args:
            ontology: 評価対象のオントロジー
            domain_concepts: ドメインの重要概念セット
            
        Returns:
            網羅性メトリクスの辞書
        """
        ontology_concepts = set(ontology.nodes())
        covered_concepts = domain_concepts.intersection(ontology_concepts)
        
        metrics = {
            "concept_coverage": len(covered_concepts) / len(domain_concepts),
            "concept_precision": len(covered_concepts) / len(ontology_concepts),
            "concept_f1": 2 * len(covered_concepts) / (len(domain_concepts) + len(ontology_concepts))
        }
        return metrics
    
    def consistency_check(self, ontology: nx.DiGraph) -> Dict[str, List[Tuple[str, str]]]:
        """
        オントロジーの一貫性をチェック
        
        Args:
            ontology: 評価対象のオントロジー
            
        Returns:
            一貫性の問題点リスト
        """
        issues = {
            "cycles": [],
            "orphan_nodes": [],
            "redundant_relations": []
        }
        
        # サイクルの検出
        try:
            cycles = list(nx.find_cycle(ontology))
            issues["cycles"] = [(edge[0], edge[1]) for edge in cycles]
        except nx.NetworkXNoCycle:
            pass
        
        # 孤立ノードの検出
        issues["orphan_nodes"] = [
            node for node in ontology.nodes()
            if ontology.in_degree(node) == 0 and ontology.out_degree(node) == 0
        ]
        
        # 冗長な関係の検出
        for node in ontology.nodes():
            ancestors = set(nx.ancestors(ontology, node))
            parents = set(ontology.predecessors(node))
            for parent in parents:
                if any(p in ancestors for p in ontology.predecessors(parent)):
                    issues["redundant_relations"].append((node, parent))
        
        return issues

class KnowledgeCompletionEvaluator:
    """知識グラフ補完の評価"""
    
    def evaluate_relation_inference(
        self,
        predicted_relations: List[Dict],
        gold_relations: List[Dict]
    ) -> Dict[str, float]:
        """
        関係推論の評価
        
        Args:
            predicted_relations: 予測された関係のリスト
            gold_relations: 正解の関係のリスト
            
        Returns:
            評価メトリクスの辞書
        """
        # 関係を文字列に変換
        pred_set = {f"{r['source']}-{r['relation']}-{r['target']}" for r in predicted_relations}
        gold_set = {f"{r['source']}-{r['relation']}-{r['target']}" for r in gold_relations}
        
        # 評価指標の計算
        true_positives = len(pred_set.intersection(gold_set))
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(gold_set) if gold_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def evaluate_link_prediction(
        self,
        predicted_links: List[Tuple[str, float]],
        actual_links: Set[Tuple[str, str]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        リンク予測の評価
        
        Args:
            predicted_links: 予測されたリンクとスコアのリスト
            actual_links: 実際のリンクのセット
            k: 評価するトップkの数
            
        Returns:
            評価メトリクスの辞書
        """
        # Hits@k
        hits_k = sum(1 for link, _ in predicted_links[:k] if link in actual_links)
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0
        for i, (link, _) in enumerate(predicted_links, 1):
            if link in actual_links:
                mrr = 1 / i
                break
        
        # Mean Average Precision (MAP)
        ap = 0
        correct = 0
        for i, (link, _) in enumerate(predicted_links, 1):
            if link in actual_links:
                correct += 1
                ap += correct / i
        map_score = ap / len(actual_links) if actual_links else 0
        
        return {
            f"hits@{k}": hits_k / k,
            "mrr": mrr,
            "map": map_score
        }

class GraphRAGEvaluator:
    """GraphRAGシステムの評価"""
    
    def evaluate_subgraph_relevance(
        self,
        query: str,
        subgraph: Dict,
        reference_concepts: Set[str]
    ) -> Dict[str, float]:
        """
        サブグラフの関連性評価
        
        Args:
            query: 検索クエリ
            subgraph: 抽出されたサブグラフ
            reference_concepts: 参照用の重要概念セット
            
        Returns:
            評価メトリクスの辞書
        """
        # 概念のカバレッジ
        extracted_concepts = {node["name"] for node in subgraph["nodes"]}
        coverage = len(extracted_concepts.intersection(reference_concepts)) / len(reference_concepts)
        
        # クエリ関連性（単純な単語の重複ベース）
        query_words = set(self._tokenize(query))
        concept_words = set(word for concept in extracted_concepts for word in self._tokenize(concept))
        word_overlap = len(query_words.intersection(concept_words)) / len(query_words) if query_words else 0
        
        # グラフの密度
        num_nodes = len(subgraph["nodes"])
        num_edges = len(subgraph["relationships"])
        density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        return {
            "concept_coverage": coverage,
            "query_relevance": word_overlap,
            "graph_density": density
        }
    
    def evaluate_answer_quality(
        self,
        generated_answer: str,
        reference_answer: str,
        context_concepts: Set[str]
    ) -> Dict[str, float]:
        """
        生成された回答の品質評価
        
        Args:
            generated_answer: 生成された回答
            reference_answer: 参照回答
            context_concepts: コンテキストの重要概念セット
            
        Returns:
            評価メトリクスの辞書
        """
        # 概念の使用率
        used_concepts = sum(1 for concept in context_concepts if concept in generated_answer)
        concept_usage = used_concepts / len(context_concepts)
        
        # 回答の具体性
        gen_words = self._tokenize(generated_answer)
        ref_words = self._tokenize(reference_answer)
        
        # 単語の重複率
        word_overlap = len(set(gen_words).intersection(set(ref_words))) / len(set(ref_words))
        
        # 回答の長さと多様性
        answer_length = len(generated_answer)
        vocabulary_size = len(set(gen_words))
        
        # キーワードの一致度
        gen_keywords = self._extract_keywords(generated_answer)
        ref_keywords = self._extract_keywords(reference_answer)
        keyword_match = len(gen_keywords.intersection(ref_keywords)) / len(ref_keywords) if ref_keywords else 0
        
        return {
            "concept_usage": concept_usage,
            "word_overlap": word_overlap,
            "answer_length": answer_length,
            "vocabulary_size": vocabulary_size,
            "keyword_match": keyword_match
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """テキストの分かち書き（簡易版）"""
        # 記号を除去し、空白で分割
        text = re.sub(r'[、。！？「」『』（）［］\s]', ' ', text)
        return [word for word in text.split() if word]
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """キーワードの抽出（簡易版）"""
        words = self._tokenize(text)
        # 出現頻度の高い単語を抽出
        word_freq = Counter(words)
        # 上位30%の単語をキーワードとして扱う
        threshold = len(word_freq) // 3
        return set(word for word, freq in word_freq.most_common(threshold))

class SystemEvaluator:
    """システム全体の評価"""
    
    def __init__(self):
        """初期化"""
        self.ontology_evaluator = OntologyEvaluator()
        self.completion_evaluator = KnowledgeCompletionEvaluator()
        self.rag_evaluator = GraphRAGEvaluator()
    
    def evaluate_pipeline(
        self,
        test_cases: List[Dict],
        ontology: nx.DiGraph,
        domain_concepts: Set[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        パイプライン全体の評価
        
        Args:
            test_cases: テストケースのリスト
            ontology: 評価対象のオントロジー
            domain_concepts: ドメインの重要概念セット
            
        Returns:
            評価結果の辞書
        """
        results = {
            "ontology": {},
            "completion": {},
            "rag": {},
            "overall": {}
        }
        
        # オントロジーの評価
        results["ontology"].update(
            self.ontology_evaluator.structural_metrics(ontology)
        )
        results["ontology"].update(
            self.ontology_evaluator.coverage_metrics(ontology, domain_concepts)
        )
        
        # テストケースごとの評価
        completion_metrics = []
        rag_metrics = []
        
        for case in test_cases:
            # 知識補完の評価
            if "relations" in case:
                rel_metrics = self.completion_evaluator.evaluate_relation_inference(
                    case["predicted_relations"],
                    case["gold_relations"]
                )
                completion_metrics.append(rel_metrics)
            
            # RAGの評価
            if "query" in case:
                rag_metric = self.rag_evaluator.evaluate_answer_quality(
                    case["generated_answer"],
                    case["reference_answer"],
                    case["context_concepts"]
                )
                rag_metrics.append(rag_metric)
        
        # 平均値の計算
        if completion_metrics:
            results["completion"] = {
                k: np.mean([m[k] for m in completion_metrics])
                for k in completion_metrics[0]
            }
        
        if rag_metrics:
            results["rag"] = {
                k: np.mean([m[k] for m in rag_metrics])
                for k in rag_metrics[0]
            }
        
        # 総合スコアの計算
        results["overall"] = {
            "ontology_score": np.mean(list(results["ontology"].values())),
            "completion_score": np.mean(list(results["completion"].values())),
            "rag_score": np.mean(list(results["rag"].values()))
        }
        results["overall"]["total_score"] = np.mean(list(results["overall"].values()))
        
        return results 
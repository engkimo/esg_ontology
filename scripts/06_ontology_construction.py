#!/usr/bin/env python3
"""
ESG関連テキストからオントロジーを構築するスクリプト
"""

import sys
from pathlib import Path
import json
import spacy
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
from dotenv import load_dotenv
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.text_processor import ESGTextProcessor
from src.knowledge_graph.ontology import ESGOntology
from src.knowledge_graph.neo4j_manager import Neo4jManager

def extract_concepts_and_relations(text_processor: ESGTextProcessor, texts: list) -> tuple:
    """
    テキストから概念と関係を抽出
    
    Returns:
        tuple: (concepts, relations)
    """
    concepts = defaultdict(set)
    relations = []
    
    for text_data in tqdm(texts, desc="テキスト解析中"):
        text = text_data["text"]
        company = text_data["company"]
        
        # エンティティと関係の抽出
        result = text_processor.process_text(text)
        
        # 概念の抽出
        for entity in result["entities"]:
            if entity["label"] in ["ORG", "PRODUCT", "EVENT", "LAW"]:
                concepts["Instance"].add((entity["text"], company))
            else:
                concepts["Concept"].add(entity["text"])
        
        # 関係の抽出
        for relation in result["relations"]:
            relations.append({
                "source": relation["source"],
                "target": relation["target"],
                "type": relation["relation"],
                "company": company
            })
    
    return dict(concepts), relations

def build_ontology(concepts: dict, relations: list) -> ESGOntology:
    """
    概念と関係からオントロジーを構築
    
    Returns:
        ESGOntology: 構築されたオントロジー
    """
    ontology = ESGOntology()
    
    # 基本概念の追加
    base_concepts = {
        "Environment": [
            "気候変動", "温室効果ガス", "再生可能エネルギー",
            "廃棄物管理", "生物多様性", "環境マネジメント"
        ],
        "Social": [
            "人権", "労働安全", "ダイバーシティ",
            "地域貢献", "サプライチェーン", "人材開発"
        ],
        "Governance": [
            "取締役会", "内部統制", "リスク管理",
            "コンプライアンス", "情報開示", "株主権利"
        ]
    }
    
    # 基本概念の登録
    for category, subconcepts in base_concepts.items():
        ontology.add_concept(category, "ESG", "is_a")
        for concept in subconcepts:
            ontology.add_concept(concept, category, "part_of")
    
    # 抽出された概念の追加
    for concept_type, items in concepts.items():
        if concept_type == "Concept":
            for concept in items:
                # 最も関連の強いカテゴリを判定
                category = ontology.classify_concept(concept)
                if category:
                    ontology.add_concept(concept, category, "related_to")
        else:  # Instance
            for instance, company in items:
                ontology.add_instance(
                    instance,
                    "Organization",
                    {"company": company}
                )
    
    # 関係の追加
    for relation in relations:
        ontology.add_relation(
            relation["source"],
            relation["target"],
            relation["type"]
        )
    
    return ontology

def export_to_neo4j(ontology: ESGOntology, neo4j: Neo4jManager) -> None:
    """
    オントロジーをNeo4jにエクスポート
    """
    # データベースのクリア
    neo4j.clear_database()
    
    # 概念の追加
    for concept, data in ontology.concepts.items():
        neo4j.add_concept(
            name=concept,
            category=data.get("category", "Other"),
            attributes=data.get("attributes", {})
        )
    
    # インスタンスの追加
    for instance, data in ontology.instances.items():
        neo4j.add_node(
            name=instance,
            category="Instance"
        )
    
    # 関係の追加
    for relation in ontology.relations:
        neo4j.add_relation(
            source=relation["source"],
            target=relation["target"],
            relation_type=relation["type"],
            properties=relation.get("properties", {})
        )

def main():
    # 環境変数の読み込み
    load_dotenv()
    
    # 抽出済みテキストの読み込み
    print("抽出済みテキストを読み込み中...")
    with open(project_root / "data/processed/esg_sections.json", "r", encoding="utf-8") as f:
        esg_sections = json.load(f)
    
    # テキストプロセッサの初期化
    print("\nテキストプロセッサを初期化中...")
    text_processor = ESGTextProcessor()
    
    # 全テキストの統合
    all_texts = []
    for category, texts in esg_sections.items():
        all_texts.extend(texts)
    
    # 概念と関係の抽出
    print("\n概念と関係を抽出中...")
    concepts, relations = extract_concepts_and_relations(text_processor, all_texts)
    
    # オントロジーの構築
    print("\nオントロジーを構築中...")
    ontology = build_ontology(concepts, relations)
    
    # Neo4jへのエクスポート
    print("\nNeo4jにエクスポート中...")
    neo4j = Neo4jManager(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    export_to_neo4j(ontology, neo4j)
    
    # 統計情報の表示
    print("\n構築結果:")
    print(f"概念数: {len(ontology.concepts)}")
    print(f"インスタンス数: {len(ontology.instances)}")
    print(f"関係数: {len(ontology.relations)}")
    
    # リソースの解放
    neo4j.close()
    print("\n処理が完了しました。")

if __name__ == "__main__":
    main() 
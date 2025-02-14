#!/usr/bin/env python3
"""
ESGドメインのテキスト処理とオントロジー構築のサンプルスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.text_processor import ESGTextProcessor
from src.knowledge_graph.ontology import ESGOntology

def main():
    # テキストプロセッサの初期化
    print("テキストプロセッサを初期化中...")
    processor = ESGTextProcessor()

    # サンプルテキスト
    sample_text = """
    当社は2050年までにカーボンニュートラルの実現を目指しています。
    温室効果ガスの削減に向けて、再生可能エネルギーの導入を進めるとともに、
    取締役会での議論を通じてESG経営を強化しています。
    また、人権や労働安全に配慮した事業運営を行い、
    コンプライアンス体制の整備にも注力しています。
    """

    print("\nテキストを処理中...")
    # テキスト処理
    result = processor.process_text(sample_text)
    
    print("\n抽出されたエンティティ:")
    for entity in result["entities"]:
        print(f"- {entity['text']} ({entity['label']})")

    print("\n抽出された関係:")
    for relation in result["relations"]:
        print(f"- {relation['source']} --[{relation['relation']}]--> {relation['target']}")

    # オントロジーの構築
    print("\nオントロジーを構築中...")
    ontology = ESGOntology()

    # 基本概念の確認
    print("\nEnvironment カテゴリの下位概念:")
    env_concepts = ontology.get_subconcepts("Environment")
    print(env_concepts)

    # 新しい概念の追加
    print("\n新しい概念を追加中...")
    ontology.add_concept(
        "排出量取引",
        "気候変動",
        relation="part_of"
    )

    # インスタンスの追加
    ontology.add_instance(
        "Jクレジット制度",
        "排出量取引",
        attributes={
            "country": "日本",
            "established": 2013
        }
    )

    # オントロジーの保存
    output_path = project_root / "data" / "processed" / "esg_ontology.json"
    print(f"\nオントロジーを保存中: {output_path}")
    ontology.save(output_path)
    print("処理が完了しました。")

if __name__ == "__main__":
    main() 
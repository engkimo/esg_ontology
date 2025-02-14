#!/usr/bin/env python3
"""
有価証券報告書からESG関連情報を抽出するスクリプト
"""

import sys
from pathlib import Path
import pandas as pd
import spacy
from tqdm import tqdm
import json
import re

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.text_processor import ESGTextProcessor
from src.knowledge_graph.ontology import ESGOntology
from src.knowledge_graph.neo4j_manager import Neo4jManager

def clean_text(text: str) -> str:
    """テキストのクリーニング"""
    if not isinstance(text, str):
        return ""
    
    # 改行とタブの正規化
    text = re.sub(r'[\n\t]+', ' ', text)
    # 連続する空白の削除
    text = re.sub(r'\s+', ' ', text)
    # 全角数字を半角に変換
    text = text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    return text.strip()

def extract_esg_sections(df: pd.DataFrame) -> dict:
    """
    ESG関連セクションの抽出
    
    Returns:
        dict: カテゴリごとのテキストデータ
    """
    esg_sections = {
        "Environment": [],
        "Social": [],
        "Governance": []
    }
    
    # 環境関連のキーワード
    env_keywords = [
        "環境", "気候変動", "カーボンニュートラル", "温室効果ガス",
        "再生可能エネルギー", "廃棄物", "リサイクル", "生物多様性"
    ]
    
    # 社会関連のキーワード
    social_keywords = [
        "人権", "労働", "安全衛生", "ダイバーシティ", "地域社会",
        "サプライチェーン", "製品安全", "情報セキュリティ"
    ]
    
    # ガバナンス関連のキーワード
    gov_keywords = [
        "コーポレートガバナンス", "内部統制", "コンプライアンス",
        "リスク管理", "取締役会", "監査", "株主"
    ]
    
    # テキストカラムの処理
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="テキスト処理中"):
        for col in text_columns:
            text = clean_text(row[col])
            if not text:
                continue
            
            # 環境関連の抽出
            if any(kw in text for kw in env_keywords):
                esg_sections["Environment"].append({
                    "text": text,
                    "source": col,
                    "company": row.get("company_name", "Unknown")
                })
            
            # 社会関連の抽出
            if any(kw in text for kw in social_keywords):
                esg_sections["Social"].append({
                    "text": text,
                    "source": col,
                    "company": row.get("company_name", "Unknown")
                })
            
            # ガバナンス関連の抽出
            if any(kw in text for kw in gov_keywords):
                esg_sections["Governance"].append({
                    "text": text,
                    "source": col,
                    "company": row.get("company_name", "Unknown")
                })
    
    return esg_sections

def main():
    # CSVファイルの読み込み
    print("CSVファイルを読み込み中...")
    df = pd.read_csv(
        project_root / "data/raw/2024_Financial_Annual_Reports.csv",
        escapechar="\\",
        quotechar='"',
        encoding="utf-8"
    )
    
    # ESG関連セクションの抽出
    print("\nESG関連セクションを抽出中...")
    esg_sections = extract_esg_sections(df)
    
    # 結果の保存
    output_dir = project_root / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "esg_sections.json", "w", encoding="utf-8") as f:
        json.dump(esg_sections, f, ensure_ascii=False, indent=2)
    
    # 統計情報の表示
    print("\n抽出結果:")
    for category, sections in esg_sections.items():
        print(f"{category}: {len(sections)}件")

if __name__ == "__main__":
    main() 
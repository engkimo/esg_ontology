# ESG オントロジー自動構築プロジェクト

## 概要

このプロジェクトは、ESG（Environmental, Social, and Governance）ドメインの知識グラフを自動構築するためのツールセットです。最新の機械学習技術を活用して、テキストデータから概念間の関係を抽出し、Neo4jデータベースに格納します。また、GraphRAGを用いて生成AIの応答品質を向上させることを目指しています。

## 主な機能

- テキストからのエンティティ・関係抽出（REBEL、spaCyなど）
- 知識グラフの構築・管理（Neo4j）
- GraphRAGによる知識補完
- GNNを用いたリンク予測

## 実装状況

### 1. テキスト処理モジュール (`src/data/text_processor.py`)
- ✅ ESGドメイン固有の語彙の登録と認識
- ✅ spaCyを使用したエンティティ抽出
- ✅ 依存関係に基づく関係抽出
- ✅ バッチ処理機能の実装

### 2. オントロジー管理モジュール (`src/knowledge_graph/ontology.py`)
- ✅ ESGドメインの基本概念階層の定義
- ✅ NetworkXを使用したグラフ構造の実装
- ✅ 概念とインスタンスの追加機能
- ✅ JSON形式でのオントロジーの保存と読み込み

### 3. 今後の実装予定
- 🔲 Neo4jとの統合（`02_neo4j_integration.py`）
- 🔲 知識グラフ補完機能（`03_knowledge_completion.py`）
- 🔲 GraphRAGシステム（`04_graph_rag.py`）

## 必要条件

- Python 3.9以上
- Neo4j 5.x
- CUDA対応GPUを推奨（Transformersモデル実行用）

## セットアップ

1. リポジトリのクローン:
```bash
git clone https://github.com/yourusername/esg_ontology.git
cd esg_ontology
```

2. 依存パッケージのインストール:
```bash
uv venv
source .venv/bin/activate  # Linuxの場合
.venv\Scripts\activate     # Windowsの場合
uv pip install -e .
```

3. spaCyの日本語モデルのインストール:
```bash
uv run python -m spacy download ja_core_news_lg
```

4. 環境変数の設定:
`.env`ファイルを作成し、以下の内容を設定してください：
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## 使用方法

### テキスト処理とオントロジー構築

```python
from src.data.text_processor import ESGTextProcessor
from src.knowledge_graph.ontology import ESGOntology

# テキストプロセッサの初期化
processor = ESGTextProcessor()

# テキストの処理
text = """
当社は2050年までにカーボンニュートラルの実現を目指しています。
温室効果ガスの削減に向けて、再生可能エネルギーの導入を進めるとともに、
取締役会での議論を通じてESG経営を強化しています。
"""
result = processor.process_text(text)

# オントロジーの構築
ontology = ESGOntology()
ontology.add_concept("排出量取引", "気候変動", relation="part_of")
ontology.add_instance("Jクレジット制度", "排出量取引", {"country": "日本", "established": 2013})
```

## プロジェクト構造

```
esg_ontology/
├── src/
│   ├── data/              # データ処理モジュール
│   ├── models/            # 機械学習モデル
│   ├── knowledge_graph/   # 知識グラフ関連
│   └── utils/             # ユーティリティ関数
├── tests/                 # テストコード
├── notebooks/            # Jupyter notebooks
├── data/                 # データファイル（.gitignore）
│   ├── raw/             # 生データ
│   └── processed/       # 処理済みデータ
├── pyproject.toml       # プロジェクト設定
└── README.md
```

## ライセンス

MITライセンス

## 貢献

プルリクエストは歓迎します。大きな変更を加える場合は、まずissueを作成して変更内容を議論してください。 
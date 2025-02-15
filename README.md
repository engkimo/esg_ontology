# ESG オントロジー自動構築プロジェクト

## 概要

このプロジェクトは、ESG（Environmental, Social, and Governance）ドメインの知識グラフを自動構築するためのツールセットです。最新の機械学習技術を活用して、テキストデータから概念間の関係を抽出し、Neo4jデータベースに格納します。また、GraphRAGを用いて生成AIの応答品質を向上させることを目指しています。

## システム構成

### 1. データ入力
- ESGテキストデータ（サステナビリティレポートなど）
- 外部知識ソース（既存のESG関連データベース）

### 2. テキスト処理モジュール (`src/data/text_processor.py`)
- ✅ ESGドメイン固有の語彙の登録と認識
- ✅ spaCyを使用したエンティティ抽出
- ✅ 依存関係に基づく関係抽出
- ✅ バッチ処理機能の実装

### 3. オントロジー管理システム
#### 3.1 知識グラフ補完 (`src/models/knowledge_completion.py`)
- ✅ LLMを使用した関係推論
  - 日本語モデル（rinna/japanese-gpt-neox-small）の統合
  - プロンプトテンプレートの実装
- ✅ GNNによるリンク予測
  - カスタムGNNモデルの実装
  - ネガティブサンプリング機能
- ✅ MPSバックエンドのサポート
  - デバイス検出と自動転送
  - バッチ処理の最適化

#### 3.2 Neo4j統合 (`src/knowledge_graph/neo4j_manager.py`)
- ✅ Neo4jデータベースとの接続管理
- ✅ インデックスの自動作成
- ✅ オントロジーのインポート機能
- ✅ 関係検索と概念検索機能

#### 3.3 出力形式 (`src/knowledge_graph/ontology.py`)
- ✅ ESGドメインの基本概念階層の定義
- ✅ NetworkXを使用したグラフ構造の実装
- ✅ 概念とインスタンスの追加機能
- ✅ JSON形式でのオントロジーの保存と読み込み

### 4. GraphRAGシステム (`src/rag/graph_rag.py`)
- ✅ サブグラフ検索
  - 類似度ベースの検索
  - カテゴリ重み付け
  - コミュニティ検出
- ✅ ハイブリッド検索
  - テキスト類似度とグラフ構造の組み合わせ
  - PageRankによる重要度計算
  - コンテキスト関連性スコアリング
- ✅ 回答生成
  - サブグラフを考慮したプロンプト生成
  - 文脈を考慮した回答生成

### 5. 評価システム (`src/evaluation/metrics.py`)
- ✅ オントロジー評価
  - 構造的メトリクス
  - カバレッジ評価
  - 一貫性チェック
- ✅ 知識補完評価
  - 関係推論の精度評価
  - リンク予測の性能評価
- ✅ RAG評価
  - サブグラフ関連性評価
  - 回答品質評価
  - コンテキストカバレッジ

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

### 3. Neo4j統合モジュール (`src/knowledge_graph/neo4j_manager.py`)
- ✅ Neo4jデータベースとの接続管理
- ✅ インデックスの自動作成
- ✅ オントロジーのインポート機能
- ✅ 関係検索と概念検索機能

### 4. 知識グラフ補完モジュール (`src/models/knowledge_completion.py`)
- ✅ LLMを使用した関係推論
- ✅ GNNによるリンク予測
- ✅ MPSバックエンドのサポート
- ✅ テストケースの実装

### 5. 今後の実装予定
- 🔲 GraphRAGシステム（`src/rag/graph_rag.py`）
- 🔲 評価指標の実装
- 🔲 ユーザーインターフェースの改善

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

### 知識グラフ補完の使用

```python
from src.models.knowledge_completion import ESGKnowledgeCompletion

# 知識補完モデルの初期化
model = ESGKnowledgeCompletion()

# LLMによる関係推論
relations = model.infer_relations_with_llm("気候変動")

# GNNによるリンク予測
predictions = model.predict_links(graph, "気候変動", top_k=5)
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
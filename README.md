# ESG オントロジー自動構築プロジェクト

## 概要

このプロジェクトは、ESG（Environmental, Social, and Governance）ドメインの知識グラフを自動構築するためのツールセットです。最新の機械学習技術を活用して、テキストデータから概念間の関係を抽出し、Neo4jデータベースに格納します。また、GraphRAGを用いて生成AIの応答品質を向上させることを目指しています。

## 主な機能

- テキストからのエンティティ・関係抽出（REBEL、spaCyなど）
- 知識グラフの構築・管理（Neo4j）
- GraphRAGによる知識補完
- GNNを用いたリンク予測

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

3. 環境変数の設定:
`.env`ファイルを作成し、以下の内容を設定してください：
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## 使用方法

詳細な使用方法は各モジュールのドキュメントを参照してください。

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
├── pyproject.toml       # プロジェクト設定
└── README.md
```

## ライセンス

MITライセンス

## 貢献

プルリクエストは歓迎します。大きな変更を加える場合は、まずissueを作成して変更内容を議論してください。 
# toolbox-abci

ABCI環境向けのツールボックスです。Conda環境の構築やvLLMサーバーの起動に必要なスクリプトを提供します。

## 概要

このリポジトリには以下のツールが含まれています：

- **install_miniconda.sh** - Minicondaのインストールスクリプト
- **create_env.sh** - 汎用Conda環境構築スクリプト
- **vllm-serve/** - vLLMサーバー関連のスクリプト群

## クイックスタート

### 1. Minicondaのインストール

```bash
cd $HOME/toolbox-abci
bash ./install_miniconda.sh
source ~/.bashrc
```

### 2. Conda環境の作成

```bash
cd $HOME/toolbox-abci
source ./create_env.sh <環境名> [Pythonバージョン]
```

例：
```bash
source ./create_env.sh my_env 3.12
```

### 3. vLLM環境の構築（オプション）

```bash
cd $HOME/toolbox-abci/vllm-serve/env_create
source ./create_env.sh
```

## ディレクトリ構成

```
toolbox-abci/
├── README.md                 # このファイル
├── install_miniconda.sh      # Minicondaインストールスクリプト
├── create_env.sh             # 汎用Conda環境構築スクリプト
├── LICENSE
└── vllm-serve/               # vLLMサーバー関連
    ├── env_create/           # vLLM環境構築スクリプト
    │   ├── create_env.sh     # vLLM環境構築メインスクリプト
    │   ├── config.yaml       # 設定ファイル
    │   ├── README.md         # vLLM環境構築の詳細ドキュメント
    │   └── ...
    └── *_server.sh           # 各モデル用サーバー起動スクリプト
```

## スクリプト詳細

### install_miniconda.sh

Minicondaを`$HOME/miniconda3`にインストールします。

**機能：**
- CPUアーキテクチャの自動判別（x86_64/aarch64）
- 既存インストールの検出とスキップ
- `.bashrc`の自動バックアップ
- `conda init`による自動設定
- Anaconda利用規約への同意

**使い方：**
```bash
bash ./install_miniconda.sh
source ~/.bashrc
```

### create_env.sh

汎用のConda環境を作成します。CUDA、cuDNN、NCCLモジュールを自動でロードします。

**機能：**
- CUDA 12.8、cuDNN 9.13、NCCL 2.28のモジュールロード
- 指定したPythonバージョンでConda環境を作成
- activate/deactivateスクリプトの自動生成

**使い方：**
```bash
source ./create_env.sh <環境名> [Pythonバージョン]
```

**引数：**
- `<環境名>` - 作成する環境の名前（必須）
- `[Pythonバージョン]` - Pythonバージョン（デフォルト: 3.12）

**環境パス：**
```
$HOME/envs/<環境名>
```

**例：**
```bash
# Python 3.12でmy_envを作成
source ./create_env.sh my_env 3.12

# 環境をアクティベート
conda activate $HOME/envs/my_env
```

### vllm-serve/env_create/create_env.sh

vLLMサーバー用の環境を構築します。PyTorch、vLLM、FlashInferなどを自動インストールします。

詳細は [vllm-serve/env_create/README.md](vllm-serve/env_create/README.md) を参照してください。

## 環境パス

すべての環境は以下のディレクトリに作成されます：

```
$HOME/envs/
├── <環境名1>/
├── <環境名2>/
└── ...
```

## 必要要件

- NVIDIA GPU（vLLM使用時）
- Environment Modulesシステム
- wget（Minicondaインストール用）
- bash

## トラブルシューティング

### condaコマンドが見つからない

```bash
source ~/.bashrc
```

または、新しいターミナルを開いてください。

### モジュールが見つからない

利用可能なモジュールを確認：
```bash
module avail cuda
module avail cudnn
module avail nccl
```

### 環境の削除

```bash
rm -rf $HOME/envs/<環境名>
```

## 更新履歴

- 2026-01-01: Minicondaをローカルインストールに変更、環境パスを$HOME/envsに統一

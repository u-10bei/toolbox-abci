# vLLM環境構築スクリプト（マルチモデル対応版）

vLLM serveを実行するために必要なパッケージをインストールする環境構築スクリプトです。

## 対応モデル

- MiniMaxAI/MiniMax-M2
- zai-org/GLM-4.6
- deepseek-ai/DeepSeek-V3.2-Exp
- moonshotai/Kimi-K2-Instruct-0905
- Qwen/Qwen3-235B-A22B-Instruct-2507
- その他vLLM対応モデル

## 概要

このスクリプトは、以下のパッケージをインストールします：

- **Python 3.12**
- **PyTorch** (CUDA対応)
- **vLLM** (0.12.0)
- **FlashInfer** (0.5.2)
- **DeepGEMM** (オプション、DeepSeekモデル用)
- **transformers** (4.48.0以上)
- **accelerate** (0.27.0以上)

## 必要要件

### システム要件
- NVIDIA GPU (H100推奨)
- NVIDIA Driver 570.133.20 以上
- CUDA 12.8
- Python 3 (PyYAMLモジュール必要)
- Environment Modulesシステム

### 事前準備

1. **Minicondaのインストール**（必須）:
   ```bash
   cd $HOME/toolbox-abci
   bash ./install_miniconda.sh
   source ~/.bashrc
   ```

2. PyYAMLのインストール（まだの場合）:
   ```bash
   pip3 install pyyaml
   ```

3. 利用可能なCUDAバージョンの確認:
   ```bash
   module avail cuda
   nvidia-smi  # ドライバーがサポートする最大CUDAバージョンを確認
   ```

## 使い方

### 基本的な使い方

デフォルト設定（`config.yaml`）を使用して環境を構築：

```bash
cd $HOME/toolbox-abci/vllm-serve/env_create
source ./create_env.sh
```

### カスタム設定を使用

独自の設定ファイルを作成して使用：

```bash
cp config.yaml my_config.yaml
# my_config.yaml を編集
source ./create_env.sh my_config.yaml
```

## 設定ファイル（config.yaml）

```yaml
environment:
  name: vllm_serve              # 環境名
  python_version: "3.12"        # Pythonバージョン

cuda:
  version: "12.8"               # CUDAバージョン
  module_name: "cuda/12.8"      # CUDAモジュール名

packages:
  pytorch:
    version: "2.8.0"            # PyTorchバージョン
  vllm:
    version: "0.12.0"           # vLLMバージョン
  flashinfer:
    version: "0.5.2"            # FlashInferバージョン
  deepgemm:
    enabled: true               # DeepGEMMを有効化（DeepSeekモデル用）
    git_url: "https://github.com/deepseek-ai/DeepGEMM.git"
    version: "v2.1.0"
```

### 設定のカスタマイズ

#### CUDA バージョンの指定

```yaml
cuda:
  version: "12.8"
  module_name: "cuda/12.8"
```

#### DeepGEMM の有効化/無効化

DeepSeekモデルを使用する場合は有効化：
```yaml
packages:
  deepgemm:
    enabled: true
```

その他のモデルのみを使用する場合は無効化可能：
```yaml
packages:
  deepgemm:
    enabled: false
```

## 環境パス

環境は以下のパスに作成されます：

```
$HOME/envs/<環境名>
```

例: `$HOME/envs/vllm_serve_2512`

## 環境の使用

### 環境のアクティベート

```bash
conda activate $HOME/envs/<環境名>
```

例：
```bash
conda activate $HOME/envs/vllm_serve_2512
```

### vLLMの実行

```bash
vllm serve <model-name>
```

例：
```bash
# 基本的な起動
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Qwen3モデル
vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507

# DeepSeekモデル（DeepGEMM有効時）
vllm serve deepseek-ai/DeepSeek-V3.2-Exp
```

### 環境のディアクティベート

```bash
conda deactivate
```

### 環境の削除

```bash
rm -rf $HOME/envs/<環境名>
```

## トラブルシューティング

### Minicondaが見つからない

```bash
cd $HOME/toolbox-abci
bash ./install_miniconda.sh
source ~/.bashrc
```

### PyYAMLが見つからない

```bash
pip3 install pyyaml
```

### CUDAモジュールが見つからない

利用可能なCUDAバージョンを確認：
```bash
module avail cuda
```

config.yamlで正しいモジュール名を指定：
```yaml
cuda:
  version: "12.8"
  module_name: "cuda/12.8"
```

### PyTorchがCUDAを認識しない

環境をアクティベート後、確認：
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

### vLLMのインポートエラー

1. PyTorchとvLLMのバージョン互換性を確認
2. FlashInferが正しくインストールされているか確認：
   ```bash
   python -c "import flashinfer"
   ```

## スクリプトの特徴

### インストール処理
- **ローカルMiniconda使用**: `$HOME/miniconda3`にインストールされたMinicondaを使用
- **CUDA Toolkit**: Conda経由でCUDA Toolkitをインストール
- **環境変数スクリプト**: activate/deactivate時に自動で環境変数を設定
- **CUDAシンボリックリンク**: 環境内でCUDAライブラリへの適切なパスを自動設定
- **検証機能**: インストール後にPyTorch、vLLM、FlashInferの動作を確認

### インストールされるもの
1. Python 3.12
2. CUDA Toolkit 12.8
3. PyTorch (CUDA対応)
4. vLLM 0.12.0
5. FlashInfer 0.5.2
6. DeepGEMM (オプション)
7. transformers
8. accelerate
9. ray
10. hf-transfer

## ファイル構成

```
env_create/
├── create_env.sh              # メインスクリプト
├── config.yaml                # 設定ファイル
├── compatibility_matrix.yaml  # 互換性情報
├── requirements.txt           # 依存パッケージ一覧
└── README.md                  # このファイル
```

## 参考情報

- **vLLM公式ドキュメント**: https://docs.vllm.ai/
- **PyTorch公式サイト**: https://pytorch.org/
- **互換性マトリクス**: `compatibility_matrix.yaml` を参照

## 更新履歴

- 2026-01-01: Minicondaをローカルインストール（$HOME/miniconda3）に変更、環境パスを$HOME/envsに変更
- 2025-12: vLLM 0.12.0、FlashInfer 0.5.2、DeepGEMMサポート追加
- 2025-11-13: FlashInfer 0.4.1に更新（vLLM 0.11.0互換性確保）
- 2025-11: 必要最小限版に簡略化、vLLM 0.11.0対応

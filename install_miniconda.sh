#!/bin/sh

# --- 改善点1: エラー時にスクリプトを即座に停止 ---
set -e

echo "Minicondaのインストールを開始します..."

# 念のためSSH等が故障したときなどに備えて~/.bashrcをバックアップしておく。
if [ -f ~/.bashrc ]; then
    cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    echo ".bashrcをバックアップしました: ~/.bashrc.backup.*"
fi

# condaのインストール先ディレクトリを定義
INSTALL_DIR=~/miniconda3

# --- 改善点5: インストール済みのチェック ---
if [ -d "${INSTALL_DIR}/bin" ] && [ -x "${INSTALL_DIR}/bin/conda" ]; then
    echo "Minicondaは既に '${INSTALL_DIR}' にインストールされています。処理をスキップします。"
    exit 0
fi

# 念のためSSH等が故障したときなどに備えて~/.bashrcをバックアップしておく。
if [ -f ~/.bashrc ]; then
    cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    echo ".bashrcをバックアップしました: ~/.bashrc.backup.*"
fi

# インストール先ディレクトリを作成。
mkdir -p ${INSTALL_DIR}

# --- 改善点4: 依存コマンドのチェック ---
if ! command -v wget &> /dev/null; then
    echo "エラー: wgetコマンドが見つかりません。wgetをインストールしてください。"
    exit 1
fi

# --- 改善点3: CPUアーキテクチャの自動判別 ---
ARCH=$(uname -m)
case "${ARCH}" in
    x86_64)
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        ;;
    aarch64)
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        ;;
    *)
        echo "エラー: サポートされていないアーキテクチャです: ${ARCH}"
        exit 1
        ;;
esac

echo "アーキテクチャ '${ARCH}' 用のインストーラーをダウンロードします..."
# Miniconda3のインストールスクリプトをダウンロードして実行。
wget ${MINICONDA_URL} -O ${INSTALL_DIR}/miniconda.sh
bash ${INSTALL_DIR}/miniconda.sh -b -u -p ${INSTALL_DIR}

# インストールしたcondaを有効化。
source ${INSTALL_DIR}/etc/profile.d/conda.sh

# --- 改善点2: conda init を使用してシェルを設定 ---
echo "conda init を実行してシェル設定を初期化します..."
conda init bash
# zshを使っている場合は conda init zsh のように変更

# --- 改善点6: Anaconda利用規約への同意 ---
echo "Anacondaの利用規約に同意します..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# インストール後の後処理
echo "Condaの更新とクリーンアップを実行します..."
conda update -n base conda -y             # Conda自体の更新
conda clean --all -y                      # 不要なキャッシュの削除
rm ${INSTALL_DIR}/miniconda.sh            # インストーラーの削除

echo "==== インストール完了 ===="
conda --version
echo "Minicondaのインストールが完了しました。"
echo "設定を反映させるために、新しいターミナルを開くか、'source ~/.bashrc' を実行してください。"
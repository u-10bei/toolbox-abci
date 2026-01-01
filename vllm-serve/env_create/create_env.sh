#!/bin/bash
#
# vLLM環境構築スクリプト（マルチモデル対応版）
#
# 対応モデル:
#   - MiniMaxAI/MiniMax-M2
#   - zai-org/GLM-4.6
#   - deepseek-ai/DeepSeek-V3.2-Exp
#   - moonshotai/Kimi-K2-Instruct-0905
#   - Qwen/Qwen3-235B-A22B-Instruct-2507
#   - その他既存モデル
#
# [使い方]
# source ./create_env.sh [config_file]
#
# [例]
# source ./create_env.sh                    # config.yaml を使用（デフォルト）
# source ./create_env.sh my_config.yaml     # カスタム設定ファイルを使用
#

set -euo pipefail

#======================================================================
# グローバル設定
#======================================================================

# スクリプトディレクトリの取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# デフォルト設定ファイル（既に定義されている場合はスキップ）
if [[ -z "${DEFAULT_CONFIG_FILE:-}" ]]; then
    readonly DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
fi

# 一時ファイル用ディレクトリ
TMP_DIR="/tmp/vllm_env_setup_$$"

#======================================================================
# 色定義
#======================================================================

if [[ -z "${C_RESET:-}" ]]; then
    if [[ -t 1 ]]; then
        C_RESET='\033[0m'
        C_RED='\033[0;31m'
        C_GREEN='\033[0;32m'
        C_YELLOW='\033[0;33m'
        C_BLUE='\033[0;34m'
        C_CYAN='\033[0;36m'
        C_BOLD='\033[1m'
    else
        C_RESET='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_CYAN='' C_BOLD=''
    fi
fi

#======================================================================
# クリーンアップ関数
#======================================================================

cleanup() {
    [[ -d "$TMP_DIR" ]] && rm -rf "$TMP_DIR"
}

trap cleanup EXIT

#======================================================================
# ユーティリティ関数
#======================================================================

# ログ出力関数
log() {
    local level=$1
    shift
    local message="$*"

    case $level in
        INFO)    echo -e "${C_BLUE}[INFO] ${message}${C_RESET}" ;;
        SUCCESS) echo -e "${C_GREEN}[SUCCESS] ${message}${C_RESET}" ;;
        WARN)    echo -e "${C_YELLOW}[WARN] ${message}${C_RESET}" ;;
        ERROR)   echo -e "${C_RED}[ERROR] ${message}${C_RESET}" >&2 ;;
    esac
}

# ヘッダー表示
print_header() {
    echo ""
    echo -e "${C_CYAN}${C_BOLD}=====================================================================${C_RESET}"
    echo -e "${C_CYAN}${C_BOLD} $1 ${C_RESET}"
    echo -e "${C_CYAN}${C_BOLD}=====================================================================${C_RESET}"
    echo ""
}

#======================================================================
# YAML読み込み関数
#======================================================================

# YAMLから値を読み取る関数
read_yaml() {
    local yaml_file=$1
    local key=$2
    local default_value=${3:-}

    [[ ! -f "$yaml_file" ]] && { echo "$default_value"; return 1; }

    python3 -c "
import yaml
import sys
import os

try:
    with open('$yaml_file', 'r') as f:
        config = yaml.safe_load(f)

    keys = '$key'.split('.')
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            print(os.path.expandvars('$default_value'))
            sys.exit(0)

    if value is None:
        print(os.path.expandvars('$default_value'))
    elif isinstance(value, bool):
        print('true' if value else 'false')
    else:
        # 環境変数を展開（\$HOME, \${HOME} など）
        print(os.path.expandvars(str(value)))
except:
    print(os.path.expandvars('$default_value'))
    sys.exit(1)
" 2>/dev/null
}

#======================================================================
# 設定のロード
#======================================================================

load_config() {
    local config_file=$1

    print_header "設定の読み込み"

    # 設定ファイルの検証
    if [[ ! -f "$config_file" ]]; then
        log ERROR "設定ファイルが見つかりません: $config_file"
        return 1
    fi

    # PyYAMLの確認
    if ! python3 -c "import yaml" 2>/dev/null; then
        log ERROR "Pythonのyamlモジュールが必要です: pip3 install pyyaml"
        return 1
    fi

    # 設定の読み込み
    ENV_NAME=$(read_yaml "$config_file" "environment.name" "vllm_serve")
    PYTHON_VERSION=$(read_yaml "$config_file" "environment.python_version" "3.12")
    CUDA_VERSION=$(read_yaml "$config_file" "cuda.version" "auto")
    CUDA_MODULE=$(read_yaml "$config_file" "cuda.module_name" "auto")

    PYTORCH_VERSION=$(read_yaml "$config_file" "packages.pytorch.version" "2.8.0")
    VLLM_VERSION=$(read_yaml "$config_file" "packages.vllm.version" "0.11.2")
    FLASHINFER_VERSION=$(read_yaml "$config_file" "packages.flashinfer.version" "0.5.2")

    # DeepGEMM設定（DeepSeekモデル用）
    DEEPGEMM_ENABLED=$(read_yaml "$config_file" "packages.deepgemm.enabled" "false")
    DEEPGEMM_GIT_URL=$(read_yaml "$config_file" "packages.deepgemm.git_url" "https://github.com/deepseek-ai/DeepGEMM.git")
    DEEPGEMM_VERSION=$(read_yaml "$config_file" "packages.deepgemm.version" "1.0.0")

    ENV_PATH="$HOME/envs/$ENV_NAME"

    # 設定の表示
    log INFO "環境名: $ENV_NAME"
    log INFO "環境パス: $ENV_PATH"
    log INFO "Pythonバージョン: $PYTHON_VERSION"
    log INFO "CUDAモジュール: $CUDA_MODULE"
    log INFO "PyTorchバージョン: $PYTORCH_VERSION"
    log INFO "vLLMバージョン: $VLLM_VERSION"
    log INFO "FlashInferバージョン: $FLASHINFER_VERSION"
    log INFO "DeepGEMM: $DEEPGEMM_ENABLED"

    log SUCCESS "設定の読み込み完了"
    return 0
}

#======================================================================
# モジュールロード
#======================================================================

load_modules() {
    print_header "環境モジュールのロード"

    # モジュールのクリア
    module purge 2>/dev/null || true

    # CUDAモジュールのロード
    log INFO "CUDAモジュール ($CUDA_MODULE) をロード中..."
    if ! module load "$CUDA_MODULE" 2>/dev/null; then
        log ERROR "CUDAモジュール $CUDA_MODULE のロードに失敗しました"
        log INFO "利用可能なCUDAモジュール:"
        module avail cuda 2>&1 | grep -i cuda || true
        return 1
    fi

    # Miniconda (ローカルインストール)
    local miniconda_local="$HOME/miniconda3"
    log INFO "Miniconda をロード中..."
    if [[ -f "$miniconda_local/etc/profile.d/conda.sh" ]]; then
        source "$miniconda_local/etc/profile.d/conda.sh"
        log SUCCESS "ローカルのMiniconda ($miniconda_local) をロードしました"
    else
        log ERROR "Minicondaが見つかりません。install_miniconda.sh を実行してください"
        return 1
    fi

    log SUCCESS "モジュールロード完了"
    return 0
}

#======================================================================
# Conda初期化
#======================================================================

initialize_conda() {
    print_header "Conda環境の初期化"

    log INFO "Condaを確認中..."

    # モジュールロード後、condaが利用可能か確認
    if ! command -v conda &>/dev/null; then
        log ERROR "condaコマンドが見つかりません"
        return 1
    fi

    log SUCCESS "Conda初期化完了: $(conda --version 2>/dev/null)"
    return 0
}

#======================================================================
# Conda環境作成
#======================================================================

create_conda_env() {
    print_header "Conda環境の作成"

    # 既存環境のチェック
    if [[ -d "$ENV_PATH" ]]; then
        log WARN "環境は既に存在します: $ENV_PATH"
        log WARN "既存の環境を使用します（削除する場合は手動で: rm -rf $ENV_PATH）"
        return 0
    fi

    # 親ディレクトリの作成
    mkdir -p "$HOME/envs"

    log INFO "Conda環境を作成中（数分かかる場合があります）..."
    log INFO "Python $PYTHON_VERSION をインストール..."

    if ! conda create --prefix "$ENV_PATH" python="$PYTHON_VERSION" -y -q 2>&1 | tee "$TMP_DIR/conda_create.log" >/dev/null; then
        log ERROR "Conda環境の作成に失敗しました（ログ: $TMP_DIR/conda_create.log）"
        return 1
    fi

    log SUCCESS "Conda環境作成完了: $ENV_PATH"
    return 0
}

#======================================================================
# CUDA Toolkitのインストール
#======================================================================

install_cuda_toolkit() {
    print_header "CUDA Toolkitのインストール"

    log INFO "CUDA Toolkit $CUDA_VERSION をインストール中..."
    log INFO "（vLLMのコンパイルにCUDAヘッダーが必要です）"

    if ! conda install --prefix "$ENV_PATH" -y -q -c nvidia cuda-toolkit="$CUDA_VERSION" 2>&1 | tee "$TMP_DIR/cuda_toolkit.log" >/dev/null; then
        log ERROR "CUDA Toolkitのインストールに失敗しました（ログ: $TMP_DIR/cuda_toolkit.log）"
        return 1
    fi

    log SUCCESS "CUDA Toolkitのインストール完了"

    # nvccのシンボリックリンクを作成（FlashInferのコンパイルに必要）
    log INFO "nvccのシンボリックリンクを作成中..."
    mkdir -p "$ENV_PATH/etc/bin"
    ln -sf "$ENV_PATH/bin/nvcc" "$ENV_PATH/etc/bin/nvcc"

    if [[ -L "$ENV_PATH/etc/bin/nvcc" ]]; then
        log SUCCESS "nvccのシンボリックリンク作成完了"
    else
        log WARN "nvccのシンボリックリンク作成に失敗しましたが、処理を続行します"
    fi

    return 0
}

#======================================================================
# パッケージインストール
#======================================================================

install_pytorch() {
    print_header "PyTorchのインストール"

    local cuda_major cuda_minor pytorch_cuda
    cuda_major=$(echo "$CUDA_VERSION" | cut -d. -f1)
    cuda_minor=$(echo "$CUDA_VERSION" | cut -d. -f2)
    pytorch_cuda="cu${cuda_major}${cuda_minor}"

    log INFO "PyTorch ${PYTORCH_VERSION}+${pytorch_cuda} をインストール中..."

    local pip_bin="$ENV_PATH/bin/pip"
    local install_cmd="$pip_bin install torch==${PYTORCH_VERSION}+${pytorch_cuda} --index-url https://download.pytorch.org/whl/${pytorch_cuda} --extra-index-url https://pypi.org/simple"

    if ! eval "$install_cmd" 2>&1 | tee "$TMP_DIR/pytorch_install.log" >/dev/null; then
        log ERROR "PyTorchのインストールに失敗しました（ログ: $TMP_DIR/pytorch_install.log）"
        return 1
    fi

    log SUCCESS "PyTorchのインストール完了"
    return 0
}

install_vllm() {
    print_header "vLLMと依存パッケージのインストール"

    log INFO "vLLM ==${VLLM_VERSION}、FlashInfer ${FLASHINFER_VERSION} をインストール中..."

    local pip_bin="$ENV_PATH/bin/pip"
    local packages=(
        "vllm==${VLLM_VERSION}"
        "ray[default]"
        "transformers>=4.48.0"
        "accelerate>=0.27.0"
        "hf-transfer"
        "blobfile"
    )
    # flashinfer-pythonはvLLMの依存関係として自動インストールされる

    if ! "$pip_bin" install "${packages[@]}" 2>&1 | tee "$TMP_DIR/vllm_install.log" >/dev/null; then
        log ERROR "vLLMのインストールに失敗しました（ログ: $TMP_DIR/vllm_install.log）"
        return 1
    fi

    log SUCCESS "vLLMのインストール完了"
    return 0
}

install_deepgemm() {
    # DeepGEMMが無効の場合はスキップ
    if [[ "$DEEPGEMM_ENABLED" != "true" ]]; then
        log INFO "DeepGEMMはスキップされます（無効化されています）"
        return 0
    fi

    print_header "DeepGEMMのインストール（DeepSeekモデル用）"

    local pip_bin="$ENV_PATH/bin/pip"

    log INFO "DeepGEMM ${DEEPGEMM_VERSION} をインストール中..."
    log INFO "リポジトリ: ${DEEPGEMM_GIT_URL}"

    # DeepGEMMをgitからインストール
    if ! "$pip_bin" install "git+${DEEPGEMM_GIT_URL}@${DEEPGEMM_VERSION}" --no-build-isolation 2>&1 | tee "$TMP_DIR/deepgemm_install.log" >/dev/null; then
        log WARN "DeepGEMMのインストールに失敗しました（ログ: $TMP_DIR/deepgemm_install.log）"
        log WARN "DeepSeekモデル以外は影響ありません。処理を続行します。"
        return 0
    fi

    log SUCCESS "DeepGEMMのインストール完了"
    return 0
}

#======================================================================
# 環境変数スクリプト作成
#======================================================================

create_env_scripts() {
    print_header "環境変数スクリプトの作成"

    local activate_dir="$ENV_PATH/etc/conda/activate.d"
    local activate_script="$activate_dir/env_vars.sh"
    mkdir -p "$activate_dir"

    # GPU検出からTORCH_CUDA_ARCH_LISTを決定（H100 = 9.0）
    local torch_cuda_arch_list="9.0"

    cat > "$activate_script" << 'EOF'
#!/bin/bash
# vLLM環境のアクティベーション時に実行されるスクリプト

# Conda環境のパスを取得
CONDA_ENV_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 元の環境変数を保存
export ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export ORIGINAL_C_INCLUDE_PATH="${C_INCLUDE_PATH:-}"
export ORIGINAL_CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}"
export ORIGINAL_PATH="${PATH:-}"

# PATHの設定（nvvm/binをciccのために追加）
export PATH="${CONDA_ENV_PATH}/nvvm/bin:${PATH}"

# LD_LIBRARY_PATHの設定
export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:/usr/lib64:/usr/lib:${LD_LIBRARY_PATH}"

# CUDAヘッダーファイルのパス設定（vLLMのコンパイルに必要）
export C_INCLUDE_PATH="${CONDA_ENV_PATH}/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CONDA_ENV_PATH}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

# vLLM最適化設定
export VLLM_DISABLE_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TORCH_CUDA_ARCH_LIST="9.0"

echo "[vLLM環境] 環境変数が設定されました"
EOF

    chmod +x "$activate_script"

    # deactivateスクリプト
    local deactivate_dir="$ENV_PATH/etc/conda/deactivate.d"
    local deactivate_script="$deactivate_dir/env_vars.sh"
    mkdir -p "$deactivate_dir"

    cat > "$deactivate_script" << 'EOF'
#!/bin/bash
# vLLM環境のディアクティベーション時に実行されるスクリプト

# 元の環境変数を復元
export LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH:-}"
export C_INCLUDE_PATH="${ORIGINAL_C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${ORIGINAL_CPLUS_INCLUDE_PATH:-}"
export PATH="${ORIGINAL_PATH:-}"

# 環境変数をクリア
unset VLLM_DISABLE_USAGE_STATS PYTORCH_CUDA_ALLOC_CONF TORCH_CUDA_ARCH_LIST
unset ORIGINAL_LD_LIBRARY_PATH ORIGINAL_C_INCLUDE_PATH ORIGINAL_CPLUS_INCLUDE_PATH ORIGINAL_PATH

echo "[vLLM環境] 環境変数がリセットされました"
EOF

    chmod +x "$deactivate_script"

    log SUCCESS "環境変数スクリプト作成完了"
    return 0
}

#======================================================================
# CUDAライブラリのシンボリックリンク作成
#======================================================================

setup_cuda_symlinks() {
    print_header "CUDAライブラリのシンボリックリンク設定"

    # lib64ディレクトリを作成
    local lib64_dir="$ENV_PATH/lib64"
    mkdir -p "$lib64_dir"

    # targets/x86_64-linux/libからlib64へのシンボリックリンク
    local target_lib="$ENV_PATH/targets/x86_64-linux/lib"

    if [[ -d "$target_lib" ]]; then
        log INFO "CUDAライブラリのシンボリックリンクを作成中..."

        # libcudart.soのシンボリックリンク
        if [[ -f "$target_lib/libcudart.so" ]]; then
            ln -sf "$target_lib/libcudart.so" "$lib64_dir/libcudart.so"
            log INFO "  libcudart.so -> linked"
        fi

        # libcuda.soのシンボリックリンク（stubsから）
        if [[ -f "$target_lib/stubs/libcuda.so" ]]; then
            mkdir -p "$lib64_dir/stubs"
            ln -sf "$target_lib/stubs/libcuda.so" "$lib64_dir/stubs/libcuda.so"
            log INFO "  libcuda.so (stub) -> linked"
        fi

        log SUCCESS "CUDAライブラリのシンボリックリンク作成完了"
    else
        log WARN "targets/x86_64-linux/lib が見つかりません"
    fi

    return 0
}

#======================================================================
# 環境検証
#======================================================================

verify_installation() {
    print_header "インストールの検証"

    local python_bin="$ENV_PATH/bin/python"
    local verification_failed=false

    # Pythonバージョン確認
    log INFO "Pythonバージョン: $("$python_bin" --version 2>&1)"

    # PyTorchの確認
    log INFO "PyTorchをテスト中..."
    if "$python_bin" -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null; then
        log SUCCESS "PyTorch: 正常"
    else
        log ERROR "PyTorchのインポートに失敗しました"
        verification_failed=true
    fi

    # vLLMの確認
    log INFO "vLLMをテスト中..."
    if "$python_bin" -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null; then
        log SUCCESS "vLLM: 正常"
    else
        log ERROR "vLLMのインポートに失敗しました"
        verification_failed=true
    fi

    # FlashInferの確認
    log INFO "FlashInferをテスト中..."
    if "$python_bin" -c "import flashinfer; print(f'FlashInfer: インポート成功')" 2>/dev/null; then
        log SUCCESS "FlashInfer: 正常"
    else
        log WARN "FlashInferのインポートに失敗しました（警告のみ）"
    fi

    if [[ "$verification_failed" == true ]]; then
        log ERROR "検証に失敗しました"
        return 1
    fi

    log SUCCESS "全ての検証に合格しました"
    return 0
}

#======================================================================
# メイン処理
#======================================================================

main() {
    local start_time
    start_time=$(date +%s)

    mkdir -p "$TMP_DIR"

    print_header "vLLM環境構築スクリプト（マルチモデル対応版）"

    # 設定ファイルの取得
    CONFIG_FILE=${1:-$DEFAULT_CONFIG_FILE}

    # 設定のロード
    if ! load_config "$CONFIG_FILE"; then
        return 1
    fi

    # モジュールのロード
    if ! load_modules; then
        return 1
    fi

    # Condaの初期化
    if ! initialize_conda; then
        return 1
    fi

    # Conda環境の作成
    if ! create_conda_env; then
        return 1
    fi

    # CUDA Toolkitのインストール
    if ! install_cuda_toolkit; then
        return 1
    fi

    # PyTorchのインストール
    if ! install_pytorch; then
        return 1
    fi

    # vLLMのインストール
    if ! install_vllm; then
        return 1
    fi

    # DeepGEMMのインストール（オプション）
    install_deepgemm

    # 環境変数スクリプトの作成
    if ! create_env_scripts; then
        return 1
    fi

    # CUDAライブラリのシンボリックリンク設定
    if ! setup_cuda_symlinks; then
        log WARN "シンボリックリンクの作成で問題が発生しました"
    fi

    # インストールの検証
    if ! verify_installation; then
        log WARN "検証で問題が発生しましたが、環境は作成されました"
    fi

    # 完了
    print_header "セットアップ完了"

    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))

    log SUCCESS "環境のセットアップが完了しました (所要時間: ${mins}分${secs}秒)"
    log INFO "環境パス: $ENV_PATH"

    # 使用方法の案内
    echo ""
    echo "環境をアクティベートするには:"
    echo "  conda activate $ENV_PATH"
    echo ""
    echo "vLLMを実行するには:"
    echo "  vllm serve <model-name>"
    echo ""
    echo "環境をディアクティベートするには:"
    echo "  conda deactivate"
}

# スクリプト実行
main "$@"

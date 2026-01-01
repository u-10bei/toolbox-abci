#!/bin/bash -l
#
# Conda環境を生成し、環境変数を設定するスクリプト。
# ロギング等のユーティリティ関数を内包しています。
#
# [使い方]
# source ./create_env.sh <env_name> [python_version]
#
# [例]
# source ./create_env.sh my_env 3.10
#

#======================================================================
# ユーティリティ関数
#======================================================================

# --- 色設定 ---
readonly C_RESET='\033[0m'
readonly C_RED='\033[0;31m'
readonly C_GREEN='\033[0;32m'
readonly C_YELLOW='\033[0;33m'
readonly C_BLUE='\033[0;34m'
readonly C_CYAN='\033[0;36m'
readonly C_BOLD='\033[1m'

# --- ログ出力関数 ---
_log_with_timestamp() {
    local log_level=$1
    local color_code=$2
    local message=$3
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    # エラー以外はstdout、エラーはstderrに出力
    if [ "$log_level" == "ERROR" ]; then
        echo -e "${color_code}[${timestamp}] ${log_level}: ${message}${C_RESET}" >&2
    else
        echo -e "${color_code}[${timestamp}] ${log_level}: ${message}${C_RESET}"
    fi
}

log_info()    { _log_with_timestamp "INFO"    "${C_BLUE}"   "$1"; }
log_success() { _log_with_timestamp "SUCCESS" "${C_GREEN}"  "$1"; }
log_warn()    { _log_with_timestamp "WARNING" "${C_YELLOW}" "$1"; }
log_error()   { _log_with_timestamp "ERROR"   "${C_RED}"    "$1"; }

print_header() {
    echo -e "\n${C_CYAN}${C_BOLD}=======================================================================${C_RESET}"
    echo -e "${C_CYAN}${C_BOLD} $1 ${C_RESET}"
    echo -e "${C_CYAN}${C_BOLD}=======================================================================${C_RESET}"
}

# --- タイマー機能（ネスト対応） ---
export __TIMER_STACK__=()

start_timer() {
    __TIMER_STACK__+=($(date +%s))
    if [ -n "$1" ]; then
        # メッセージがある場合はINFOログとして出力
        log_info "$1"
    fi
}

end_timer() {
    local end_time
    end_time=$(date +%s)
    if [ ${#__TIMER_STACK__[@]} -gt 0 ]; then
        # スタックの最後の要素（最新の開始時間）を取得
        local start_time=${__TIMER_STACK__[-1]}

        # スタックの最後の要素を削除する
        if ((BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then
             unset '__TIMER_STACK__[-1]'
        else # Fallback for older bash versions
             __TIMER_STACK__=("${__TIMER_STACK__[@]:0:(${#__TIMER_STACK__[@]} - 1)}")
        fi

        local elapsed=$((end_time - start_time))
        local mins=$((elapsed / 60))
        local secs=$((elapsed % 60))
        log_success "処理が完了しました (経過時間: ${mins}分${secs}秒)"
    else
        log_warn "タイマースタックが空です。end_timerが過剰に呼び出された可能性があります。"
    fi
}

# --- エラーハンドリング機能 ---
handle_error() {
    local exit_code=$1
    local line_no=$2
    local failed_command=$3
    log_error "コマンドが失敗しました (終了コード: ${exit_code}, 行番号: ${line_no})"
    log_error "失敗したコマンド: ${failed_command}"
}


#======================================================================
# メイン処理
#======================================================================

set -e

#----------------------------------------------------------------------
# モジュールのロード (CUDA, cuDNN, NCCL)
#----------------------------------------------------------------------
print_header "モジュールのロード"

# CUDA 12.8
if module load cuda/12.8/12.8.1 2>/dev/null; then
    log_success "cuda/12.8/12.8.1 をロードしました"
else
    log_error "cuda/12.8/12.8.1 のロードに失敗しました"
    return 1
fi

# cuDNN (CUDA 12.8 対応の最新版)
if module load cudnn/9.13/9.13.0 2>/dev/null; then
    log_success "cudnn/9.13/9.13.0 をロードしました"
else
    log_error "cudnn/9.13/9.13.0 のロードに失敗しました"
    return 1
fi

# NCCL (最新版)
if module load nccl/2.28/2.28.3-1 2>/dev/null; then
    log_success "nccl/2.28/2.28.3-1 をロードしました"
else
    log_error "nccl/2.28/2.28.3-1 のロードに失敗しました"
    return 1
fi

# Miniconda (ローカルインストール)
MINICONDA_LOCAL="$HOME/miniconda3"
if [ -f "$MINICONDA_LOCAL/etc/profile.d/conda.sh" ]; then
    source "$MINICONDA_LOCAL/etc/profile.d/conda.sh"
    log_success "ローカルのMiniconda ($MINICONDA_LOCAL) をロードしました"
else
    log_error "Minicondaが見つかりません。install_miniconda.sh を実行してください"
    return 1
fi

log_info "CUDA_HOME=$CUDA_HOME"

# --- 設定変数 ---
ENVNAME=${1}
if [ -z "$ENVNAME" ]; then
    log_error "環境名が指定されていません。例: source ./create_env.sh my_env 3.10"
    return 1
fi

# Pythonバージョンの引数化
PYTHON_VERSION=${2:-"3.12"} # 第2引数がなければ "3.12" をデフォルト値とする
log_info "環境名: ${ENVNAME}, Pythonバージョン: ${PYTHON_VERSION}"

# Conda のベースパスを設定
CONDA_BASE="$HOME/miniconda3"
if ! command -v conda &>/dev/null; then
    log_error "condaコマンドが見つかりません"
    return 1
fi

if [ ! -d "$CONDA_BASE" ]; then
    log_error "Condaベースディレクトリが見つかりません: $CONDA_BASE"
    return 1
fi

log_info "Conda base: $CONDA_BASE"

export CONDA_ENV_FULL_PATH="$HOME/envs/$ENVNAME"

print_header "Conda環境 '$ENVNAME' のセットアップ"

if [ ! -d "$CONDA_ENV_FULL_PATH" ]; then
    start_timer "Conda環境 '$ENVNAME' (Python ${PYTHON_VERSION}) を新規作成します..."
    if ! conda create --prefix "$CONDA_ENV_FULL_PATH" python="$PYTHON_VERSION" -y --quiet; then
        log_error "Conda環境 '$ENVNAME' の作成に失敗しました。"
        return 1
    fi
    end_timer

    log_info "アクティベート時に読み込む環境変数スクリプトを作成します..."
    ACTIVATE_DIR="$CONDA_ENV_FULL_PATH/etc/conda/activate.d"
    ACTIVATE_SCRIPT="$ACTIVATE_DIR/env_vars.sh"
    mkdir -p "$ACTIVATE_DIR"

    # Conda環境のlibパスのみを追加（CUDA関連はmoduleで設定済み）
    cat <<EOF > "$ACTIVATE_SCRIPT"
#!/bin/bash
export ORIGINAL_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="${CONDA_ENV_FULL_PATH}/lib:\$LD_LIBRARY_PATH"
EOF
    chmod +x "$ACTIVATE_SCRIPT"
    log_info "作成完了: $ACTIVATE_SCRIPT"

    # --- deactivate.d スクリプト ---
    log_info "ディアクティベート時に読み込む環境変数スクリプトを作成します..."
    DEACTIVATE_DIR="$CONDA_ENV_FULL_PATH/etc/conda/deactivate.d"
    DEACTIVATE_SCRIPT="$DEACTIVATE_DIR/env_vars_rollback.sh"
    mkdir -p "$DEACTIVATE_DIR"

    cat <<EOF > "$DEACTIVATE_SCRIPT"
#!/bin/bash
export LD_LIBRARY_PATH="\$ORIGINAL_LD_LIBRARY_PATH"
unset ORIGINAL_LD_LIBRARY_PATH
EOF
    chmod +x "$DEACTIVATE_SCRIPT"
    log_info "作成完了: $DEACTIVATE_SCRIPT"

else
    log_warn "Conda環境 '$ENVNAME' は既に存在するため、新規作成をスキップします。"
fi

log_success "Conda環境のセットアップスクリプトが完了しました。"
log_info "次のコマンドで環境をアクティベートしてください: conda activate ${ENVNAME}"
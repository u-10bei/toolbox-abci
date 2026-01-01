#!/bin/bash
# --- Slurm ジョブ設定 ---
#SBATCH --job-name=235b_vllm
#SBATCH --partition=P08317
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=120
#SBATCH --mem=700G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "---------------------------------"
echo "--- MoE対応vLLMジョブ開始 ---"
echo "ジョブ実行ホスト: $(hostname)"
echo "現在時刻: $(date)"
echo "作業ディレクトリ: $(pwd)"
echo "---------------------------------"

# プロセスID変数を初期化
pid_vllm=""
pid_nvsmi=""
cleanup_counter=0

# 環境変数の設定
ENV_DIR="/home/matsuolab/shared/envs/vllm_serve_2512"
# ローカルで環境構築した場合
# ENV_DIR="$HOME/envs/vllm_serve_2512"

# HuggingFaceトークンをinfo.yamlから読み込み
CONFIG_FILE="/home/matsuolab/horie/vllm_serve/config/info.yaml"
if [[ -f "$CONFIG_FILE" ]]; then
    HF_TOKEN=$(grep "^HuggingFace:" "$CONFIG_FILE" | awk '{print $2}')
    if [[ -z "$HF_TOKEN" ]]; then
        echo "❌ HuggingFaceトークンの読み込みに失敗しました"
        exit 1
    fi
    echo "✅ HuggingFaceトークンを読み込みました"
else
    echo "❌ 設定ファイルが見つかりません: $CONFIG_FILE"
    exit 1
fi

MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"

# ログディレクトリの作成
mkdir -p logs

# キャッシュクリア関数の定義
clear_compile_cache() {
    echo "🧹 コンパイルキャッシュをクリア中..."
    rm -rf $HOME/.cache/vllm/torch_compile_cache 2>/dev/null || true
    rm -rf $HOME/.cache/flashinfer 2>/dev/null || true
    echo "✅ キャッシュクリア完了"
}

# クリーンアップ関数の定義
cleanup() {
    echo "🧹 クリーンアップ開始..."
    cleanup_counter=$((cleanup_counter + 1))
    
    if [[ $cleanup_counter -gt 1 ]]; then
        echo "🚨 強制終了中..."
        exit 130
    fi
    
    # VLLMプロセス終了
    if [[ -n "${pid_vllm-}" ]] && kill -0 "$pid_vllm" 2>/dev/null; then
        echo "VLLM PID $pid_vllm を終了中..."
        kill "$pid_vllm" 2>/dev/null || true
        sleep 5
        kill -9 "$pid_vllm" 2>/dev/null || true
    fi
    
    # GPU監視プロセス終了
    if [[ -n "${pid_nvsmi-}" ]] && kill -0 "$pid_nvsmi" 2>/dev/null; then
        echo "GPU監視 PID $pid_nvsmi を終了中..."
        kill "$pid_nvsmi" 2>/dev/null || true
    fi
    
    # 残りのvLLMプロセスを強制終了
    pkill -f "vllm serve" 2>/dev/null || true
    # ポート11303-11403の範囲をクリーンアップ
    for port in {11303..11403}; do
        lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done

    echo "✅ クリーンアップ完了"
    exit 0
}

# シグナルハンドラー設定
trap cleanup SIGINT SIGTERM EXIT

module purge
module load nccl/2.24.3

# リソース制限の解除（マルチGPU推論に必要）
echo "📋 リソース制限解除中..."
ulimit -l unlimited || true  # ロック可能なメモリサイズを無制限に（NCCL通信に必要）
ulimit -v unlimited || true  # 仮想メモリサイズを無制限に（大規模モデルロードに必要）
echo "✅ リソース制限解除完了"

# conda初期化（安全に実行）
echo "📋 conda初期化..."
if source /home/appli/miniconda3/24.7.1-py312/etc/profile.d/conda.sh 2>/dev/null; then
    echo "✅ conda初期化成功"
else
    echo "❌ conda初期化失敗"
    exit 1
fi

# conda環境アクティベート（安全に実行）
echo "📋 conda環境アクティベート..."
if conda activate $ENV_DIR 2>/dev/null; then
    echo "Conda env : $CONDA_DEFAULT_ENV"
    echo "Python path: $(which python)"
    echo "Pip path: $(which pip)"
    echo "✅ conda環境アクティベート成功"

    # LD_LIBRARY_PATHを明示的に再設定（conda環境のlibを最優先）
    export LD_LIBRARY_PATH="$ENV_DIR/lib:$LD_LIBRARY_PATH"

    # CUDAインクルードパスを明示的に設定
    export CPATH="$ENV_DIR/targets/x86_64-linux/include:$CPATH"
    export C_INCLUDE_PATH="$ENV_DIR/targets/x86_64-linux/include:$C_INCLUDE_PATH"
    export CPLUS_INCLUDE_PATH="$ENV_DIR/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH"
    echo "📋 CUDA include paths set"
else
    echo "❌ conda環境アクティベート失敗"
    exit 1
fi

# --- MoE設定ファイル確認 ---
MOE_CONFIG_DIR="$ENV_DIR/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs"
CONFIG_FILE="$MOE_CONFIG_DIR/E=32,N=1536,device_name=NVIDIA_H100_80GB_HBM3.json"

if [[ -f "$CONFIG_FILE" ]]; then
    echo "MoE設定ファイル確認済み"
else
    echo "MoE設定ファイルを作成中..."
    mkdir -p "$MOE_CONFIG_DIR"
    cat > "$CONFIG_FILE" << 'EOF'
{
  "64": {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 8,
    "num_stages": 4
  },
  "128": {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 3
  }
}
EOF
    chmod 644 "$CONFIG_FILE"
    echo "MoE設定ファイル作成完了"
fi

# ここでset -eを有効にする（moduleやconda処理が完了してから）
echo "📋 厳密エラーチェック有効化..."
set -euo pipefail

# 環境変数の設定
export MODEL_NAME=$MODEL_NAME
echo "Model name: $MODEL_NAME"

# 並列処理の最適化設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN 
export NCCL_TREE_THRESHOLD=0  # 通信最適化
export TORCH_NCCL_AVOID_RECORD_STREAMS=0

# OpenMP設定（CPU並列処理の最適化）
export OMP_NUM_THREADS=16  # CPUスレッド数を適切に制限（GPUあたり2スレッド）
export MKL_NUM_THREADS=16  # Intel MKLスレッド数
export NUMEXPR_NUM_THREADS=16  # NumExprスレッド数

# Hugging Face 認証と設定
export HF_TOKEN=$HF_TOKEN
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
# 注意: TRANSFORMERS_CACHEは廃止予定のため、HF_HOMEのみ使用
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"  

# vLLM最適化設定
export VLLM_DISABLE_USAGE_STATS=1  # 使用統計無効化
export VLLM_LOG_LEVEL=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # メモリ断片化対策

# CUDAアーキテクチャリストの設定
export TORCH_CUDA_ARCH_LIST="9.0"

# 既存のVLLMプロセスをクリーンアップ
echo "🧹 既存のVLLMプロセスをクリーンアップ中..."
pkill -f "vllm serve" 2>/dev/null || true
# ポート11303-11403の範囲をクリーンアップ
for port in {11303..11403}; do
    lsof -ti:$port 2>/dev/null | xargs -r kill -9 2>/dev/null || true
done
sleep 3

#--- GPU 監視 -------------------------------------------------------
echo "📊 GPU監視を開始..."
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > logs/nvidia-smi.log 2>&1 &
pid_nvsmi=$!

#--- vLLM 起動（8GPU）------------------------------------
echo "🚀 VLLM起動中（MoEモデル）..."
echo "起動時刻: $(date)"

# 既存のコンパイルキャッシュを削除して再コンパイルを強制
echo "🧹 既存のコンパイルキャッシュを削除中..."
rm -rf $HOME/.cache/vllm/torch_compile_cache
rm -rf $HOME/.cache/flashinfer
echo "✅ vLLMとFlashInferのキャッシュを削除しました"
export VLLM_LOG="logs/vllm.log"

# VLLMログファイルを初期化
> $VLLM_LOG

# 空きポート検索関数
find_available_port() {
    local start_port=${1:-11303}
    local end_port=${2:-11403}
    local port=$start_port

    while [[ $port -le $end_port ]]; do
        # lsofコマンドでポートの使用状況をチェック（より確実）
        if ! lsof -i :${port} -sTCP:LISTEN >/dev/null 2>&1; then
            echo "$port"
            return 0
        fi
        port=$((port + 1))
    done

    echo "❌ 利用可能なポートが見つかりませんでした (${start_port}-${end_port})"
    return 1
}

# 動的ポート割り当て
PORT_OUTPUT=$(find_available_port 11303 11403)
VLLM_PORT=$(echo "$PORT_OUTPUT" | tail -1)
if [[ -z "$VLLM_PORT" ]] || [[ ! "$VLLM_PORT" =~ ^[0-9]+$ ]]; then
    echo "❌ ポート割り当てに失敗しました (取得値: '$VLLM_PORT')"
    exit 1
fi

echo "📋 使用ポート: ${VLLM_PORT}"

# VLLM起動パラメータを定義
VLLM_ARGS=(
    "$MODEL_NAME"
    "--host" "0.0.0.0"
    "--port" "$VLLM_PORT"
    "--tensor-parallel-size" "4"
    "--pipeline-parallel-size" "2"
    "--gpu-memory-utilization" "0.85"  # 修正: 0.9→0.85 より安全な値
    "--max-model-len" "131072"  # 追加: 最大シーケンス長制限
    "--disable-custom-all-reduce"
    "--trust-remote-code"
    "--enable-chunked-prefill"  # 追加: チャンク化プリフィル有効化
    "--max-num-seqs" "60"  # 追加: 同時処理シーケンス数制限
)

# MoEモデル用のパラメータを条件付きで追加
if [[ "$MODEL_NAME" == *"MoE"* ]] || [[ "$MODEL_NAME" == *"moe"* ]] || [[ "$MODEL_NAME" == *"Qwen"* ]]; then
    echo "📋 MoEモデル用パラメータを追加..."
    VLLM_ARGS+=("--enable-expert-parallel")
fi

# VLLM起動
echo "🚀 VLLMコマンド実行:"
echo "vllm serve ${VLLM_ARGS[*]}"
vllm serve "${VLLM_ARGS[@]}" > $VLLM_LOG 2>&1 &
pid_vllm=$!

echo "VLLM PID: $pid_vllm"

#--- ヘルスチェック -------------------------------------------------
echo "🔍 VLLMヘルスチェック開始..."
MAX_WAIT_TIME=1800  # 修正: 30分→40分 (1800秒)
WAIT_INTERVAL=30
elapsed_time=0

while [[ $elapsed_time -lt $MAX_WAIT_TIME ]]; do
    echo "$(date +%T) [${elapsed_time}s] VLLM起動確認中..."
    
    # VLLMプロセスが生きているかチェック
    if [[ -n "${pid_vllm-}" ]] && [[ "${pid_vllm-}" != "" ]] && ! kill -0 "$pid_vllm" 2>/dev/null; then
        echo "❌ VLLMプロセスが予期せず終了しました"
        echo "=== VLLM ログの最後の50行 ==="
        tail -50 $VLLM_LOG 2>/dev/null || echo "ログファイルが見つかりません"
        echo "=========================="

        # コンパイルエラーの場合はキャッシュをクリア
        if grep -i "ninja build failed\|cicc.*not found\|flashinfer.*failed" $VLLM_LOG >/dev/null 2>&1; then
            echo "⚠️ コンパイルエラーを検出。次回起動時のためにキャッシュをクリアします。"
            clear_compile_cache
        fi
        exit 1
    fi

    # ヘルスチェック（複数アドレス試行 + ログベース判定）
    health_check_success=false

    # 方法1: vLLMログから起動完了を確認（最も確実）
    if grep -q "Starting vLLM API server.*on http://0.0.0.0:${VLLM_PORT}" $VLLM_LOG 2>/dev/null; then
        # API serverが起動していることを確認
        if grep -q "Application startup complete" $VLLM_LOG 2>/dev/null; then
            health_check_success=true
            echo "🔍 vLLMログから起動完了を検出"
        fi
    fi

    # 方法2: 複数のアドレスでcurlを試行
    if [[ "$health_check_success" == "false" ]]; then
        for addr in "127.0.0.1" "localhost" "$(hostname)"; do
            if timeout 15 curl -s --max-time 10 "http://${addr}:${VLLM_PORT}/health" >/dev/null 2>&1; then
                health_check_success=true
                echo "🔍 ヘルスチェック成功: ${addr}:${VLLM_PORT}"
                break
            fi
        done
    fi

    if [[ "$health_check_success" == "true" ]]; then
        echo "✅ VLLM READY! 起動時間: ${elapsed_time}秒"

        # モデル情報を取得（複数アドレス試行）
        echo "🔍 モデル情報取得中..."
        for addr in "127.0.0.1" "localhost" "$(hostname)"; do
            if timeout 15 curl -s "http://${addr}:${VLLM_PORT}/v1/models" | jq '.' 2>/dev/null; then
                break
            fi
        done
        break
    fi
    
    # エラーログの確認（より詳細に）
    if [[ -f $VLLM_LOG ]]; then
        # 致命的エラーをチェック
        if grep -i "fatal\|critical\|cuda error\|out of memory\|segmentation fault" $VLLM_LOG >/dev/null 2>&1; then
            echo "❌ VLLMログで致命的エラーを検出:"
            echo "=== VLLM 致命的エラーログ ==="
            grep -i "fatal\|critical\|cuda error\|out of memory\|segmentation fault" $VLLM_LOG | tail -10
            echo "===================="

            # コンパイルエラーの場合はキャッシュをクリア
            if grep -i "ninja build failed\|cicc.*not found\|flashinfer.*failed" $VLLM_LOG >/dev/null 2>&1; then
                echo "⚠️ コンパイルエラーを検出。次回起動時のためにキャッシュをクリアします。"
                clear_compile_cache
            fi
            exit 1
        fi
        
        # 一般的なエラーをチェック（警告程度）
        if grep -i "error\|exception\|failed\|traceback" $VLLM_LOG >/dev/null 2>&1; then
            echo "⚠️ VLLMログで警告を検出（継続監視中）:"
            grep -i "error\|exception\|failed\|traceback" $VLLM_LOG | tail -3
        fi
    fi
    
    # 進行状況をログから確認
    if [[ -f $VLLM_LOG ]] && grep -i "loading\|model\|initialized\|warmup" $VLLM_LOG >/dev/null 2>&1; then
        echo "📦 進行状況:"
        tail -3 $VLLM_LOG | grep -v "^$" || true
    fi
    
    sleep $WAIT_INTERVAL
    elapsed_time=$((elapsed_time + WAIT_INTERVAL))
done

# タイムアウトチェック
if [[ $elapsed_time -ge $MAX_WAIT_TIME ]]; then
    echo "❌ VLLM起動タイムアウト ${MAX_WAIT_TIME}秒"
    echo "=== VLLM ログの最後の100行 ==="
    tail -100 $VLLM_LOG 2>/dev/null || echo "ログファイルが見つかりません"
    echo "=========================="

    # コンパイルエラーの場合はキャッシュをクリア
    if grep -i "ninja build failed\|cicc.*not found\|flashinfer.*failed" $VLLM_LOG >/dev/null 2>&1; then
        echo "⚠️ コンパイルエラーを検出。次回起動時のためにキャッシュをクリアします。"
        clear_compile_cache
    fi
    exit 1
fi

# GPU使用量確認
echo "📊 GPU使用量確認:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# 簡単なテスト実行
echo "🧪 簡単な推論テスト実行..."
TEST_RESPONSE=$(timeout 60 curl -s -X POST http://127.0.0.1:${VLLM_PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "prompt": "Hello, how are you?",
    "max_tokens": 10,
    "temperature": 0.7
  }' 2>/dev/null) || echo "テスト失敗"

if [[ -n "$TEST_RESPONSE" ]] && echo "$TEST_RESPONSE" | jq '.choices[0].text' >/dev/null 2>&1; then
    echo "✅ 推論テスト成功"
    echo "Response: $(echo "$TEST_RESPONSE" | jq -r '.choices[0].text' | tr -d '\n')"
else
    echo "⚠️ 推論テスト失敗（サーバーは起動済み）"
fi

echo "🎉 vLLMサーバー起動完了！"
echo "📡 接続先: http://$(hostname):${VLLM_PORT}"
echo "🔍 ヘルスチェック: curl http://$(hostname):${VLLM_PORT}/health"
echo "📖 モデル一覧: curl http://$(hostname):${VLLM_PORT}/v1/models"

# GPU監視を停止
echo "📊 GPU監視を停止..."
if [[ -n "${pid_nvsmi-}" ]] && [[ "${pid_nvsmi-}" != "" ]]; then
    kill "$pid_nvsmi" 2>/dev/null || true
fi

# メインプロセスを待機（Ctrl+Cで終了可能）
echo "🔄 サーバーを実行中... (Ctrl+C で停止)"
trap cleanup SIGINT SIGTERM

# VLLMプロセスを待機
wait $pid_vllm
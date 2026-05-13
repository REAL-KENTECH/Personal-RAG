#!/usr/bin/env bash
# Personal RAG — 일상 실행 스크립트.
#
# 사용:
#   ./run.sh               # 기본 (127.0.0.1:8501, 로컬만)
#   ./run.sh --public      # 0.0.0.0:8501, 같은 네트워크 / 외부 노출
#   ./run.sh --port 8080   # 다른 포트
#   ./run.sh --bg          # nohup 백그라운드
#
# .env 가 있으면 자동으로 로드. setup.sh 가 만든 .venv 활성화.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# --- 옵션 파싱 -----------------------------------------------------------
ADDR="127.0.0.1"
PORT="8501"
BG=0
while [ $# -gt 0 ]; do
    case "$1" in
        --public) ADDR="0.0.0.0"; shift ;;
        --port)   PORT="$2"; shift 2 ;;
        --bg)     BG=1; shift ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
done

# --- 가상환경 ------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "✗ .venv 가 없습니다. 먼저 'bash setup.sh' 실행하세요."
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- .env 자동 로드 ------------------------------------------------------
# python-dotenv 가 app.py 안에서 처리하지만 systemd 같은 환경 아닐 때를
# 위해 셸 export 도 함께. 빈 줄/주석 스킵.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# --- 실행 ----------------------------------------------------------------
ARGS=(
    streamlit run app.py
    --server.address "$ADDR"
    --server.port "$PORT"
    --server.headless true
    --browser.gatherUsageStats false
)

echo "▸ Personal RAG starting on http://${ADDR}:${PORT}"
if [ "$BG" -eq 1 ]; then
    nohup "${ARGS[@]}" > "$ROOT_DIR/app.log" 2>&1 &
    echo "✓ PID $! — logs: tail -f app.log"
else
    exec "${ARGS[@]}"
fi

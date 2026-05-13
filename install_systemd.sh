#!/usr/bin/env bash
# Personal RAG — systemd 서비스 설치 스크립트.
#
# 효과:
#   - /etc/systemd/system/personal-rag.service 등록
#   - 부팅 시 자동 시작 (enable)
#   - 죽으면 자동 재시작 (Restart=always)
#   - journalctl 로 통합 로그 조회 가능
#
# 사용:
#   sudo bash install_systemd.sh                  # 기본 (127.0.0.1:8501)
#   sudo bash install_systemd.sh --public         # 0.0.0.0:8501
#   sudo bash install_systemd.sh --port 8080
#   sudo bash install_systemd.sh --name myrag     # 서비스 이름 변경
#
# 재실행 안전 — 기존 유닛 파일은 덮어쓰고 재시작.

set -euo pipefail

# --- 색상 ----------------------------------------------------------------
if [ -t 1 ]; then
    BOLD=$(tput bold); GREEN=$(tput setaf 2); CYAN=$(tput setaf 6); RESET=$(tput sgr0)
else
    BOLD=''; GREEN=''; CYAN=''; RESET=''
fi
say() { printf '%s▸ %s%s\n' "$CYAN$BOLD" "$1" "$RESET"; }
ok()  { printf '%s✓ %s%s\n' "$GREEN" "$1" "$RESET"; }
die() { printf '✗ %s\n' "$1" >&2; exit 1; }

# --- 옵션 ----------------------------------------------------------------
ADDR="127.0.0.1"
PORT="8501"
SVC_NAME="personal-rag"
while [ $# -gt 0 ]; do
    case "$1" in
        --public) ADDR="0.0.0.0"; shift ;;
        --port)   PORT="$2"; shift 2 ;;
        --name)   SVC_NAME="$2"; shift 2 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) die "알 수 없는 옵션: $1" ;;
    esac
done

# --- 사전 점검 -----------------------------------------------------------
if [ "$(id -u)" -ne 0 ]; then
    die "sudo 로 실행해야 합니다.  예: sudo bash $0"
fi
if ! command -v systemctl >/dev/null 2>&1; then
    die "systemd 가 없는 시스템입니다. (Docker 컨테이너 안 또는 비-Linux?)"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$ROOT_DIR/app.py" ] || die "app.py 를 찾지 못했습니다 — Personal-RAG 디렉토리에서 실행하세요."
[ -d "$ROOT_DIR/.venv" ] || die ".venv 가 없습니다. 먼저 bash setup.sh 를 실행하세요."

# 어떤 사용자로 돌릴지 — sudo 호출한 원래 유저.
RUN_USER="${SUDO_USER:-$USER}"
if [ -z "$RUN_USER" ] || [ "$RUN_USER" = "root" ]; then
    die "비-root 사용자에서 sudo 로 실행해야 RUN_USER 가 정확히 잡힙니다."
fi

ENV_FILE="$ROOT_DIR/.env"
[ -f "$ENV_FILE" ] || die ".env 가 없습니다. 먼저 만들고 키를 채우세요."

UNIT_PATH="/etc/systemd/system/${SVC_NAME}.service"
say "유닛 파일 작성: $UNIT_PATH"

cat > "$UNIT_PATH" <<EOF
[Unit]
Description=Personal RAG (Streamlit) — ${SVC_NAME}
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${ROOT_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${ROOT_DIR}/.venv/bin/streamlit run app.py \\
    --server.address ${ADDR} \\
    --server.port ${PORT} \\
    --server.headless true \\
    --browser.gatherUsageStats false
Restart=always
RestartSec=5
# 보안: 시스템 일부 차단 (앱이 시스템 디렉토리 못 건드림)
ProtectSystem=full
NoNewPrivileges=true
# 로그는 journald 가 자동 캡처

[Install]
WantedBy=multi-user.target
EOF

chmod 644 "$UNIT_PATH"
ok "유닛 파일 작성 완료"

say "systemd 데몬 reload + enable + start"
systemctl daemon-reload
systemctl enable "$SVC_NAME"
systemctl restart "$SVC_NAME"
sleep 2

# --- 결과 확인 -----------------------------------------------------------
echo
if systemctl is-active --quiet "$SVC_NAME"; then
    ok "${SVC_NAME} 서비스 동작 중"
else
    die "${SVC_NAME} 시작 실패 — sudo journalctl -u ${SVC_NAME} -n 50 으로 로그 확인하세요."
fi

echo
printf '%s%s===================== 설치 완료 =====================%s\n' "$GREEN" "$BOLD" "$RESET"
echo
echo "주소: http://${ADDR}:${PORT}"
echo "사용자: ${RUN_USER}"
echo "디렉토리: ${ROOT_DIR}"
echo
echo "자주 쓰는 명령:"
echo "  sudo systemctl status   ${SVC_NAME}"
echo "  sudo systemctl restart  ${SVC_NAME}"
echo "  sudo systemctl stop     ${SVC_NAME}"
echo "  sudo journalctl -u ${SVC_NAME} -n 200 -f      # 실시간 로그"
echo "  sudo systemctl disable  ${SVC_NAME}            # 부팅 자동시작 해제"
echo
echo "제거하려면:  sudo bash uninstall_systemd.sh --name ${SVC_NAME}"
echo

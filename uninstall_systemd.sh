#!/usr/bin/env bash
# Personal RAG — systemd 서비스 제거 스크립트.
#
# 사용:
#   sudo bash uninstall_systemd.sh             # 기본 (personal-rag)
#   sudo bash uninstall_systemd.sh --name myrag

set -euo pipefail

SVC_NAME="personal-rag"
while [ $# -gt 0 ]; do
    case "$1" in
        --name) SVC_NAME="$2"; shift 2 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "sudo 로 실행하세요." >&2
    exit 1
fi

UNIT="/etc/systemd/system/${SVC_NAME}.service"

if [ ! -f "$UNIT" ]; then
    echo "유닛 파일 없음: $UNIT — 이미 제거됐거나 다른 이름인 듯합니다."
    exit 0
fi

echo "▸ 서비스 중지 + disable + 유닛 파일 삭제: ${SVC_NAME}"
systemctl stop "$SVC_NAME" 2>/dev/null || true
systemctl disable "$SVC_NAME" 2>/dev/null || true
rm -f "$UNIT"
systemctl daemon-reload
systemctl reset-failed "$SVC_NAME" 2>/dev/null || true

echo "✓ 제거 완료. (앱 파일 / .venv / .env / 데이터 디렉토리는 그대로)"

#!/usr/bin/env bash
# Personal RAG — 새 서버 1회 셋업 스크립트.
#
# 동작 흐름:
#   1. Python 3.10+ 확인
#   2. 시스템 패키지 (tesseract OCR 등) 설치 — sudo 필요
#   3. .venv 가상환경 생성
#   4. requirements.txt + requirements-extras.txt 설치
#   5. .env 템플릿 생성 (아직 없을 때만)
#   6. 다음 단계 안내
#
# 사용:
#   bash setup.sh              # 전체 셋업
#   bash setup.sh --no-extras  # docling/pymupdf 등 무거운 패키지 스킵
#   bash setup.sh --no-system  # apt 설치 단계 스킵 (이미 깔려있거나 root 아닐 때)
#
# 재실행 안전. 이미 있는 것은 건드리지 않음.

set -euo pipefail

# --- 색상 ----------------------------------------------------------------
if [ -t 1 ]; then
    BOLD=$(tput bold); DIM=$(tput dim); RED=$(tput setaf 1)
    GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
    RESET=$(tput sgr0)
else
    BOLD=''; DIM=''; RED=''; GREEN=''; YELLOW=''; CYAN=''; RESET=''
fi
say()  { printf '%s▸ %s%s\n' "$CYAN$BOLD" "$1" "$RESET"; }
ok()   { printf '%s✓ %s%s\n' "$GREEN" "$1" "$RESET"; }
warn() { printf '%s! %s%s\n' "$YELLOW" "$1" "$RESET"; }
die()  { printf '%s✗ %s%s\n' "$RED$BOLD" "$1" "$RESET"; exit 1; }

# --- 옵션 ----------------------------------------------------------------
INSTALL_EXTRAS=1
INSTALL_SYSTEM=1
for arg in "$@"; do
    case "$arg" in
        --no-extras) INSTALL_EXTRAS=0 ;;
        --no-system) INSTALL_SYSTEM=0 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) die "알 수 없는 옵션: $arg" ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
say "작업 디렉토리: $ROOT_DIR"

# --- 1) Python 버전 ------------------------------------------------------
say "Python 3.10+ 확인"
if ! command -v python3 >/dev/null 2>&1; then
    die "python3 가 설치되어 있지 않습니다."
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    die "Python ${PY_VER} 입니다. 3.10 이상이 필요합니다."
fi
ok "Python ${PY_VER}"

# --- 2) 시스템 패키지 ----------------------------------------------------
if [ "$INSTALL_SYSTEM" -eq 1 ]; then
    say "시스템 패키지 설치 (sudo 비밀번호 필요할 수 있음)"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y
        sudo apt-get install -y \
            python3-pip python3-venv git build-essential \
            tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng
        ok "apt 패키지 완료 (tesseract OCR 포함)"
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y python3-pip git tesseract tesseract-langpack-kor
        ok "dnf 패키지 완료"
    elif command -v brew >/dev/null 2>&1; then
        brew install tesseract tesseract-lang
        ok "brew 패키지 완료 (한국어 lang pack 포함)"
    else
        warn "지원되는 패키지 매니저 (apt/dnf/brew) 를 찾지 못했습니다. 수동 설치 필요."
    fi
else
    warn "시스템 패키지 설치 단계 스킵 (--no-system)"
fi

# --- 3) 가상환경 ---------------------------------------------------------
if [ -d ".venv" ]; then
    ok ".venv 가 이미 존재합니다 — 재사용"
else
    say ".venv 가상환경 생성"
    python3 -m venv .venv
    ok "가상환경 생성 완료"
fi

# 활성화 (이 셸 안에서만)
# shellcheck disable=SC1091
source .venv/bin/activate
ok "가상환경 활성: $(python -c 'import sys; print(sys.executable)')"

# --- 4) pip 의존성 -------------------------------------------------------
say "pip 업그레이드"
pip install --quiet --upgrade pip

say "기본 의존성 설치 (requirements.txt)"
pip install --quiet -r requirements.txt
ok "기본 의존성 완료"

if [ "$INSTALL_EXTRAS" -eq 1 ] && [ -f requirements-extras.txt ]; then
    say "추가 의존성 설치 (requirements-extras.txt — docling / pymupdf / tavily 등)"
    pip install --quiet -r requirements-extras.txt
    ok "추가 의존성 완료"
else
    warn "추가 의존성 스킵 — Docling 같은 고품질 파서는 비활성됩니다."
fi

# --- 5) .env 템플릿 ------------------------------------------------------
if [ -f .env ]; then
    ok ".env 가 이미 존재합니다 — 덮어쓰지 않음"
else
    say ".env 템플릿 생성"
    cat > .env <<'ENVEOF'
# Personal RAG — 환경 변수. 이 파일은 git 에 commit 하지 마세요 (.gitignore 됨).

# LLM API
HF_TOKEN=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
FIREWORKS_API_KEY=
# DASHSCOPE_API_KEY=

# Supabase (영속 로깅 + 사용자 인증 + pgvector 다 합쳐 한 곳)
SUPABASE_URL=
SUPABASE_KEY=

# 웹 검색 (선택)
# TAVILY_API_KEY=
# BRAVE_API_KEY=
ENVEOF
    chmod 600 .env
    ok ".env 생성. 키 입력 후 다시 실행하세요."
fi

# --- 6) 안내 -------------------------------------------------------------
echo
printf '%s%s===================== 셋업 완료 =====================%s\n' "$GREEN" "$BOLD" "$RESET"
echo
echo "다음 단계:"
echo
printf '  %s1.%s .env 의 키를 채워주세요:\n' "$BOLD" "$RESET"
printf '       %s$ nano .env%s\n\n' "$DIM" "$RESET"
printf '  %s2.%s 앱 실행 (간단):\n' "$BOLD" "$RESET"
printf '       %s$ ./run.sh%s\n' "$DIM" "$RESET"
printf '       %s$ bash run.sh --public  # 외부 0.0.0.0 노출%s\n\n' "$DIM" "$RESET"
printf '  %s3.%s 운영 (자동 재시작 + 도메인 + HTTPS) 은 README 의 systemd / nginx 섹션 참고\n\n' "$BOLD" "$RESET"

# Supabase 스키마 안내 (한 번에 본 사람이 못 보면 헤맴)
echo "Supabase 처음 사용이라면 SQL Editor 에서 다음 4개 파일을 차례로 실행:"
echo "  • db_schema.sql           (chat_turns / agent_runs / events)"
echo "  • db_schema_pgvector.sql  (청크 임베딩 영속)"
echo "  • db_schema_users.sql     (회원가입 / 로그인)"
echo "  • db_schema_preferences.sql (API 키 / 설정 동기화)"
echo "  • db_schema_sessions.sql  (대화 목록 복원)"
echo

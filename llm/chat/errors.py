"""Provider-aware LLM error → human-readable Korean explainer.

Pattern-matches the most common failure modes (HF Inference Providers
permission missing, gated model, invalid token, quota, OpenAI Responses-
API-only model, network) and surfaces an actionable message. Falls back
to the raw error text otherwise. Every failure is also appended to
events.jsonl for offline analysis.
"""

import streamlit as st

from auth.users import _log_event
from llm.params import _is_fireworks_endpoint


def _show_llm_error(e: Exception):
    """Render a Korean, actionable error message for common LLM call failures."""
    err_str = str(e) or ''
    el = err_str.lower()

    # Persist the error to events.jsonl so failures aren't only visible
    # in the live UI. Best-effort — never raises.
    try:
        _log_event('llm_error', {
            'provider': st.session_state.get('provider', ''),
            'model': st.session_state.get('model', ''),
            'base_url': st.session_state.get('base_url', ''),
            'exception_type': type(e).__name__,
            'error': err_str[:1500],
        })
    except Exception:
        pass

    def show(headline, body_md):
        st.error(headline)
        st.markdown(body_md)
        with st.expander('원본 오류 메시지'):
            st.code(err_str[:1500] or repr(e))

    # HF Router — Inference Providers permission missing (403)
    if ('inference providers' in el
            and ('insufficient permissions' in el or 'does not have' in el
                 or 'this authentication method' in el)):
        show(
            'Hugging Face 토큰에 "Inference Providers" 권한이 없습니다.',
            """
**해결 방법:**

1. https://huggingface.co/settings/tokens 접속
2. 사용 중인 토큰 이름 클릭 → **Edit permissions** (또는 + Create new token → Fine-grained)
3. 다음 권한 체크:
   - ✅ **Make calls to Inference Providers** ← 필수
   - ✅ Make calls to the serverless Inference API (권장)
   - ✅ Read access to public repositories (자동 포함)
   - Llama / Gemma 같은 gated 모델 쓰면 → Read access to selected gated repositories 추가
4. **Save** → 설정 탭에서 Hugging Face 토큰 칸 갱신 (또는 .env / Cloud secrets 의 HF_TOKEN 갱신)
5. (Cloud 배포면) Manage app → Reboot
            """,
        )
        return

    # HF — gated model (Llama, Gemma 등) 라이선스 미수락 / gated 권한 누락
    if ('gated' in el or
            ('access to model' in el and ('granted' in el or 'requires' in el)) or
            ('is restricted' in el and 'license' in el)):
        show(
            'Gated 모델 접근 권한 없음.',
            """
**해결 방법:**

1. 해당 모델 페이지 (예: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 에서 라이선스 약관 수락
2. https://huggingface.co/settings/tokens → 토큰 권한에 **Read access to selected gated repositories** 추가하고 해당 모델 체크
3. 토큰 저장 → 앱에서 다시 시도
            """,
        )
        return

    # Fireworks AI — 모델 ID 오타 또는 계정 권한 부족
    if (_is_fireworks_endpoint() and
            ('not found' in el or 'inaccessible' in el or 'not deployed' in el)):
        show(
            'Fireworks 에서 이 모델을 찾을 수 없거나 계정에 접근 권한이 없습니다.',
            """
**해결 방법:**

1. **모델 ID 오타 확인** — Fireworks 모델 ID 는 항상 `accounts/fireworks/models/<name>` 풀 경로. 짧은 이름만 적으면 안 됩니다.
2. **계정 등급 제한** — Llama 405B 같은 일부 대형 모델은 유료 등급 / 신청 필요. 무료 계정에서는 70B 이하 권장.
3. **권장 동작 모델:** 설정 → Fireworks AI 모델 드롭다운에서:
   - `accounts/fireworks/models/llama-v3p3-70b-instruct` (기본)
   - `accounts/fireworks/models/qwen2p5-72b-instruct` (한국어 우수)
   - `accounts/fireworks/models/deepseek-v3` (강력)
4. 현재 본인 계정에서 호출 가능한 모델 전체 목록: https://fireworks.ai/models
            """,
        )
        return

    # Provider 미지원 / 모델 deploy 안 됨
    if 'not supported by any provider' in el or 'model_not_supported' in el:
        show(
            '이 모델을 서빙하는 활성 Provider가 없습니다.',
            """
**해결 방법:**

1. 설정 탭에서 다른 모델로 변경 (한국어가 강한 추천: `Qwen/Qwen3-Next-80B-A3B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`, `deepseek-ai/DeepSeek-V4-Pro`)
2. 또는 모델 카드 우측 "Inference Providers" 박스에서 서빙 가능한 provider 확인 후 https://huggingface.co/settings/inference-providers 에서 활성화 (Together AI / Cerebras / Hyperbolic 등)
            """,
        )
        return

    # OpenAI — Responses API 전용 모델 (gpt-5-pro, o1-pro 등) → Chat Completions 거부
    if 'v1/responses' in el or 'only supported in v1/responses' in el:
        show(
            '이 모델은 OpenAI 의 Responses API 전용 — 우리 앱은 호환되지 않습니다.',
            """
**원인:** `gpt-5-pro`, `o1-pro` 같은 "Pro" 등급 일부 모델은 OpenAI 의 새 `/v1/responses` 엔드포인트로만 제공됩니다. 본 앱은 표준 `/v1/chat/completions` 를 사용해 호출 자체가 거부됩니다.

**해결: 설정 → OpenAI 모델 변경.** 다음은 Chat Completions 로 정상 동작합니다:

- `gpt-5` — Pro 의 약 7할 성능, 같은 추론
- `gpt-5-mini` — 빠르고 저렴, 일반 RAG 충분
- `gpt-4.1` — 안정적인 차세대
- `o3` — 추론 강화 (논문 분석 / 수학에 강함)
- `o4-mini` — 추론 + 빠름
            """,
        )
        return

    # 토큰 자체가 무효 / 만료
    if ('invalid' in el and 'token' in el) or 'bad credentials' in el or '401' in el:
        show(
            'API 키 / 토큰 인증 실패 (401).',
            """
**해결 방법:**

1. 설정 탭에서 현재 공급자의 API 키가 비어있지 않은지 확인
2. 토큰이 만료됐다면 발급처에서 새로 만들기:
   - Hugging Face: https://huggingface.co/settings/tokens
   - OpenAI: https://platform.openai.com/api-keys
3. 새 키로 갱신 → 다시 시도
            """,
        )
        return

    # 결제 / 쿼터 초과
    if ('quota' in el or 'rate limit' in el or 'insufficient_quota' in el
            or '429' in err_str or '402' in err_str):
        show(
            '사용량 / 쿼터 초과 또는 결제 필요.',
            """
**해결 방법:**

- OpenAI: https://platform.openai.com/account/billing 에서 결제 / 한도 확인
- HF Inference Providers: provider 별로 무료 크레딧 한도 다름. 다른 provider 활성화 시도.
- 잠시 후 재시도하거나 더 작은 모델로 변경.
            """,
        )
        return

    # 네트워크 / 타임아웃
    if 'timeout' in el or 'timed out' in el or 'connection' in el:
        show(
            '네트워크 / 타임아웃.',
            '잠시 후 다시 시도해 주세요. 문제가 지속되면 다른 공급자로 변경.',
        )
        return

    # Fallback
    st.error(f'요청 실패: {e}')

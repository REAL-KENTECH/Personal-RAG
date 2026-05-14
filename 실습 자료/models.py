"""LLM, 임베더를 백엔드에 맞춰 만들어 돌려주는 헬퍼.

.env 의 BACKEND 값에 따라 두 경로 중 하나가 선택된다.

  BACKEND=openai      유료. ChatOpenAI + OpenAIEmbeddings. 빠르고 간편.
  BACKEND=huggingface 무료. ChatHuggingFace(HF Inference API) +
                      sentence-transformers 로컬 임베더.

각 step 파일이 이 두 함수만 호출하면 backend 가 바뀌어도 코드 수정 없이
같은 흐름이 돈다.
"""

import os

from dotenv import load_dotenv

load_dotenv()

BACKEND = os.getenv("BACKEND", "openai").lower().strip()


def get_llm(temperature: float = 0.0):
    """현재 백엔드의 ChatModel 객체를 만든다."""
    if BACKEND == "huggingface":
        # HF Inference API 경유 — 무료 크레딧, HF_TOKEN 필요
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        repo_id = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        endpoint = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            # HF endpoint 는 temperature 0 을 허용하지 않으므로 ε 으로 대체
            temperature=max(temperature, 0.01),
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=endpoint)

    # 기본: OpenAI
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        temperature=temperature,
    )


def get_embeddings():
    """현재 백엔드의 Embeddings 객체를 만든다."""
    if BACKEND == "huggingface":
        # sentence-transformers 가중치를 로컬로 받아 돌린다 (오프라인 가능).
        # 첫 실행 때만 모델을 다운로드 (~470 MB). 이후엔 캐시 사용.
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=os.getenv(
                "HF_EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings()


def describe_backend() -> str:
    """현재 어떤 백엔드를 쓰는지 한 줄로 알려주는 함수 — 디버깅용."""
    if BACKEND == "huggingface":
        return (
            f"backend=huggingface | "
            f"LLM={os.getenv('HF_LLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct')} | "
            f"Embed={os.getenv('HF_EMBED_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')}"
        )
    return (
        f"backend=openai | "
        f"LLM={os.getenv('OPENAI_LLM_MODEL', 'gpt-4o-mini')} | "
        f"Embed=text-embedding-3-small"
    )

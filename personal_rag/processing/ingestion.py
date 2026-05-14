"""End-to-end ingestion: parse → chunk → embed → save → pgvector dual-write.

``ingest_files`` consumes the list of Streamlit uploaded files, skips
duplicates (by name and by content hash), wires every other module
together, and surfaces per-file progress in a Streamlit ``st.status``
container so the user can watch what's happening on large batches.

``remove_doc`` and ``reindex_all`` are the inverse / refresh operations
the Documents view exposes.
"""

import time

import streamlit as st

from ..auth.users import _log_event
from ..data.storage import (
    _pages_dir,
    _pgvector_delete_doc,
    _pgvector_upsert_doc,
    compute_doc_id,
    delete_saved_doc,
    load_doc,
    save_doc,
)
from ..llm.clients import load_embedder
from .chunking import chunk_elements, chunk_text
from .parsing import parse_file, render_pdf_pages_to_dir


def _embed_with_progress(chunks: list, embedder, status, label_prefix: str,
                          batch_size: int = 32):
    """Embed chunks in batches and surface per-batch progress to a Streamlit
    st.status container. Returns the stacked numpy embedding matrix."""
    total = len(chunks)
    if total == 0:
        import numpy as _np
        return _np.zeros((0, 1), dtype=_np.float32)
    if total <= batch_size:
        status.update(label=f'{label_prefix} 임베딩 ({total}청크)')
        return embedder.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
    import numpy as _np
    parts = []
    done = 0
    bar = st.progress(0.0, text=f'{label_prefix} 임베딩 {done}/{total} 청크')
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        parts.append(embedder.encode(
            batch, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ))
        done = min(i + batch_size, total)
        bar.progress(done / total, text=f'{label_prefix} 임베딩 {done}/{total} 청크')
    bar.empty()
    return _np.vstack(parts)


def ingest_files(files):
    """For each new file: parse (Docling for PDF) → chunk with page metadata
    → embed (batched, progress-reported) → optionally render page images
    (only if multimodal is enabled, otherwise skipped to save time)
    → save → register."""
    eid = st.session_state['embedder_model']
    size = st.session_state['chunk_size']
    overlap = st.session_state['chunk_overlap']
    render_pages = bool(st.session_state.get('include_page_images'))
    existing_ids = {d['id'] for d in st.session_state['docs']}
    existing_names = {d['name'] for d in st.session_state['docs']}
    new_count = 0
    batch_t0 = time.time()

    with st.status(f'{len(files)}개 파일 인덱싱 시작', expanded=True) as status:
        for f in files:
            if f.name in existing_names:
                status.update(label=f'{f.name}: 이미 등록됨, 건너뜀')
                _log_event('doc_skip', {'name': f.name, 'reason': 'duplicate_name'})
                continue

            file_t0 = time.time()
            status.update(label=f'{f.name}: 파싱 중...')
            parsed = parse_file(f)
            raw = parsed['raw_text']
            if not raw.strip():
                st.warning(f'{f.name}: 추출된 텍스트가 없어 인덱스에서 제외합니다.')
                _log_event('doc_skip', {'name': f.name, 'reason': 'no_text'})
                continue

            status.update(label=f'{f.name}: 청크 분할 중...')
            chunks, chunk_pages = chunk_elements(parsed['elements'], size, overlap)
            if not chunks:
                st.warning(f'{f.name}: 청크가 생성되지 않았습니다.')
                _log_event('doc_skip', {'name': f.name, 'reason': 'no_chunks'})
                continue

            did = compute_doc_id(f.name, raw, size, overlap)
            if did in existing_ids:
                _log_event('doc_skip', {'name': f.name, 'reason': 'duplicate_content', 'doc_id': did})
                continue

            cached = load_doc(eid, did)
            if cached is not None:
                status.update(label=f'{f.name}: 캐시에서 즉시 복원 ({len(cached["chunks"])} 청크)')
                doc = cached
                # Cached restore: pgvector copy may not exist on this Cloud
                # container yet (different deploy / different region). Upsert
                # is idempotent so retrying is cheap.
                _pgvector_upsert_doc(eid, doc)
            else:
                # Load embedder lazily — the very first call also downloads the
                # model weights (~470 MB for MiniLM, ~2.2 GB for BGE-M3).
                status.update(label=f'{f.name}: 임베더 준비 중 (모델 로드)...')
                embedder = load_embedder(eid)
                embs = _embed_with_progress(
                    chunks, embedder, status, label_prefix=f.name,
                )

                has_imgs = False
                if render_pages and parsed['is_pdf'] and parsed['pdf_bytes']:
                    status.update(label=f'{f.name}: 페이지 이미지 렌더 중 (멀티모달용)...')
                    try:
                        n_pages = render_pdf_pages_to_dir(
                            parsed['pdf_bytes'], _pages_dir(eid, did)
                        )
                        has_imgs = n_pages > 0
                    except Exception as e:
                        st.warning(f'{f.name}: 페이지 이미지 렌더 실패 ({e}).')

                status.update(label=f'{f.name}: 디스크 저장 중...')
                doc = {
                    'id': did, 'name': f.name, 'raw_text': raw,
                    'chunks': chunks, 'chunk_pages': chunk_pages,
                    'page_count': parsed['page_count'],
                    'has_page_images': has_imgs,
                    'is_pdf': parsed['is_pdf'],
                    'embeddings': embs,
                    'chunk_size': size, 'chunk_overlap': overlap,
                }
                save_doc(eid, doc)
                # Also push chunk embeddings to pgvector if Supabase is
                # configured — no-op otherwise, never blocks ingestion.
                _pgvector_upsert_doc(eid, doc)

            st.session_state['docs'].append(doc)
            existing_ids.add(did)
            existing_names.add(doc['name'])
            new_count += 1
            status.update(label=f'{f.name}: 완료 ({len(doc["chunks"])} 청크)')
            _log_event('doc_ingest', {
                'name': doc['name'],
                'doc_id': did,
                'n_chunks': len(doc['chunks']),
                'page_count': doc.get('page_count', 0),
                'has_page_images': doc.get('has_page_images', False),
                'is_pdf': doc.get('is_pdf', False),
                'embedder': eid,
                'chunk_size': size,
                'chunk_overlap': overlap,
                'elapsed_seconds': round(time.time() - file_t0, 3),
                'from_cache': cached is not None,
            })

        if new_count > 0:
            status.update(
                label=f'{new_count}개 새 문서 인덱싱 완료',
                state='complete', expanded=False,
            )
        else:
            status.update(
                label='새로 인덱싱된 문서 없음', state='complete', expanded=False,
            )
    _log_event('doc_ingest_batch', {
        'files_offered': len(files),
        'new_count': new_count,
        'elapsed_seconds': round(time.time() - batch_t0, 3),
    })
    return new_count


def remove_doc(doc_id: str):
    eid = st.session_state['embedder_model']
    doc_name = next(
        (d['name'] for d in st.session_state['docs'] if d['id'] == doc_id),
        '',
    )
    delete_saved_doc(eid, doc_id)
    _pgvector_delete_doc(doc_id)
    st.session_state['docs'] = [d for d in st.session_state['docs'] if d['id'] != doc_id]
    _log_event('doc_delete', {
        'doc_id': doc_id, 'name': doc_name, 'embedder': eid,
    })


def reindex_all():
    """Re-chunk + re-embed all current docs with current settings.

    Note: re-chunking uses the stored raw_text, which loses Docling page
    boundaries. Page metadata is preserved only for original ingestion.
    """
    eid = st.session_state['embedder_model']
    size = st.session_state['chunk_size']
    overlap = st.session_state['chunk_overlap']
    new_docs = []
    embedder = load_embedder(eid)
    for d in st.session_state['docs']:
        raw = d.get('raw_text', '')
        if not raw:
            new_docs.append(d)
            continue
        chunks = chunk_text(raw, size, overlap)
        if not chunks:
            continue
        new_id = compute_doc_id(d['name'], raw, size, overlap)
        cached = load_doc(eid, new_id)
        if cached is not None:
            new_docs.append(cached)
            continue
        embs = embedder.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        nd = {
            'id': new_id, 'name': d['name'], 'raw_text': raw,
            'chunks': chunks,
            'chunk_pages': [[] for _ in chunks],
            'page_count': d.get('page_count', 0),
            'has_page_images': d.get('has_page_images', False),
            'is_pdf': d.get('is_pdf', False),
            'embeddings': embs,
            'chunk_size': size, 'chunk_overlap': overlap,
        }
        save_doc(eid, nd)
        new_docs.append(nd)
    st.session_state['docs'] = new_docs

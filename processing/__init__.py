"""Document processing pipeline: parse → chunk → embed → persist.

- ``parsing`` accepts a Streamlit uploaded file and returns a uniform
  ``{raw_text, elements, page_count, is_pdf, pdf_bytes}`` dict regardless of
  source format (PDF with Docling/pypdf/OCR fallbacks, DOCX, CSV, HWPX,
  plain text). Page numbers are preserved when the source carries them.
- ``chunking`` turns the parsed elements into ~``chunk_size``-character
  chunks, splitting paragraphs on sentence boundaries (Korean endings
  included) so chunks don't end mid-sentence.
- ``ingestion`` wires parsing + chunking + embedding + disk save +
  pgvector dual-write together and surfaces progress in the Streamlit UI.
"""

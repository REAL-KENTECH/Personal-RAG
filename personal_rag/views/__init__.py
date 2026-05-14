"""View renderers — one Streamlit "page" per file, selected by the sidebar.

Each ``view_*`` function is the body of a Streamlit tab. The router in
app.py picks one based on ``st.session_state['active_view']`` and calls
it; everything else (sidebar, page_config, CSS) runs around it on every
rerun. Views read state freely and never persist it themselves — the
chat / agent / ingestion helpers are responsible for that.
"""

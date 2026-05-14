"""Authentication, per-user filesystem layout, and persistent preferences.

- ``supabase_io`` wraps every Supabase RPC + table call (signup/login/prefs/insert).
- ``users`` owns the on-disk per-user directory tree, the event log, and the
  Streamlit login gate that runs before any view code.
- ``prefs`` persists slowly-changing UI selections (provider, API keys,
  retrieval knobs, current session) so a Cloud idle reset doesn't wipe them.
"""

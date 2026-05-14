"""Task-templated agent workflows (email / report / summary / analysis / comparison).

Each task in ``AGENT_TASKS`` declares its input form, retrieval query
formula, and message builder. ``run_agent_task`` wires the chosen
template to the retrieval pipeline, the LLM call, and the audit log so
the view layer only has to render the form and display the output.
"""

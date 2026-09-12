from viz.components.tokenizer import component as tokenizer_component

# Every visualizable component registers itself here. Add new entries as
# more parts of llm_from_scratch grow a web UI.
COMPONENTS = [tokenizer_component]

__all__ = ["COMPONENTS"]

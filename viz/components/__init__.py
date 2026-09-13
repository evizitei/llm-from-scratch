from viz.components.tokenizer import component as tokenizer_component
from viz.components.vocabulary import component as vocabulary_component

# Every visualizable component registers itself here. Add new entries as
# more parts of llm_from_scratch grow a web UI.
COMPONENTS = [tokenizer_component, vocabulary_component]

__all__ = ["COMPONENTS"]

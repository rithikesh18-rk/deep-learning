"""
Alias for src.vector_db to maintain backwards compatibility.
"""
from src.vector_db import (
    build_vector_store,
    save_index,
    load_index,
    get_embedding_model,
    DEFAULT_EMBEDDING_MODEL
)

# Backwards compatible alias
create_vector_store = build_vector_store
save_vector_store = save_index
load_vector_store = load_index

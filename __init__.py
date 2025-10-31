# __init__.py for HuggingFace Trending Models Node
# This file makes the directory a valid ComfyUI custom node module.

from .hf_trending_models_node import (
    HuggingFaceTrendingModelsNode,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS
)

# Required by ComfyUI to detect custom nodes
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]

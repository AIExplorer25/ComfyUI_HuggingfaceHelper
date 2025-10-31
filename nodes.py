import requests
import os
from typing import List
from comfy.model_base import ComfyNode, ComfyUIOutput

class HuggingFaceTrendingModelsNode(ComfyNode):
    """Fetch trending Hugging Face models and output links."""
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("model_links",)
    FUNCTION = "fetch_models"
    CATEGORY = "HuggingFace"

    def fetch_models(self) -> ComfyUIOutput:
        base_url = "https://huggingface.co/models-json"
        all_models = []
        output_file = os.path.join(os.path.dirname(__file__), "trending_models.txt")

        for page in range(0, 11):  # pages 0 to 10
            url = f"{base_url}?p={page}&sort=trending&withCount=true"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for model in data.get("models", []):
                        model_id = model.get("id")
                        if model_id:
                            link = f"https://huggingface.co/{model_id}"
                            all_models.append(link)
                else:
                    print(f"Failed to fetch page {page}: {resp.status_code}")
            except Exception as e:
                print(f"Error on page {page}: {e}")

        # Write all model IDs to a text file
        with open(output_file, "w", encoding="utf-8") as f:
            for link in all_models:
                f.write(link + "\n")

        print(f"✅ Saved {len(all_models)} model links to {output_file}")
        return (all_models,)

NODE_CLASS_MAPPINGS = {
    "HuggingFaceTrendingModels": HuggingFaceTrendingModelsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuggingFaceTrendingModels": "🤗 Fetch HuggingFace Trending Models"
}

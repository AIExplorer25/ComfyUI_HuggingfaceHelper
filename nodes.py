import requests
import os
from typing import List
import os
import re
from huggingface_hub import ModelCard, model_info
import json
class HuggingFacePromptBuilder:
    """
    🧠 Custom node to combine Hugging Face model card info and a user instruction.
    Example use:
      - Connect output of 🤗 HuggingFace Model Card Reader
      - Enter custom instruction (e.g. "Summarize this model card in 3 sentences")
      - Output: formatted prompt text
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_card_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text from HuggingFace Model Card Reader node"
                }),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": """You are an expert summarizer for AI model cards on Hugging Face.

                                Summarize the following model card in 3–5 sentences.
                                Focus only on:
                                - What this model does
                                - Its key features or purpose
                                - Any special capabilities or limitations
                                - Its typical use cases
                                
                                Avoid technical installation details, example code, and author credits.
                                
                                Model Card:
                                {{model_card_text}}""",
                    "tooltip": "Enter your custom instruction or task for the LLM"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_text",)
    FUNCTION = "build_prompt"
    CATEGORY = "🤗 HuggingFace Utils"

    def build_prompt(self, model_card_text, instruction):
        # Combine the instruction and model card content into one coherent prompt
        combined_prompt = (
            f"{instruction.strip()}\n\n"
            f"---\n\n"
            f"Model Card:\n{model_card_text.strip()}"
        )

        return (combined_prompt,)
class HuggingFaceTrendingModelsNode:
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

    def fetch_models(self):
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
                            all_models.append(model_id)
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




class HuggingFaceModelCardReaderNode:
    """Read a repo ID from a text file and fetch its Hugging Face model card."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "multiline": False,
                    "default": "custom_nodes/trending_models.txt",
                    "tooltip": "Path to .txt file containing repo IDs (one per line)"
                }),
                "line_no": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Current line number (auto increments after each run)"
                }),
                "repo_id": ("STRING", {
                    "multiline": False,
                    "default": "repoid",
                    "tooltip": "Enter repo id or file path"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("model_name", "last_modified", "tags", "model_card_text", "next_line_no")
    FUNCTION = "fetch_model_card"
    CATEGORY = "HuggingFace"
    
    def clean_html(self, raw_html: str) -> str:
        """Remove HTML tags and extra whitespace from model card content."""
        clean = re.sub(r"<.*?>", "", raw_html, flags=re.DOTALL)  # strip tags
        clean = re.sub(r"&[a-zA-Z#0-9]+;", "", clean)  # remove HTML entities
        clean = re.sub(r"\s+", " ", clean).strip()  # collapse whitespace
        return clean
        
    def fetch_model_card(self, file_path: str, line_no: int, repo_id: str ):
        """Fetch model card info for the repo ID at the given line."""
        #if not os.path.exists(file_path):
            #raise FileNotFoundError(f"❌ File not found: {file_path}")
        lines = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
    
            if line_no < 1 or line_no > len(lines):
                raise ValueError(f"❌ Line number {line_no} is out of range (file has {len(lines)} lines).")
    
            repo_id = lines[line_no - 1]
        else:
            repo_id = repo_id
        print(f"📦 Fetching model card for: {repo_id}")

        try:
            info = model_info(repo_id)
            card = ModelCard.load(repo_id)
        except Exception as e:
            raise RuntimeError(f"⚠️ Failed to fetch model card for {repo_id}: {e}")

        model_name = info.modelId
        last_modified = str(info.lastModified)
        tags = ", ".join(info.tags) if info.tags else "None"
        model_card_text = self.clean_html(card.text)  # limit for display, optional
        if lines:
            next_line = line_no + 1 if line_no < len(lines) else 1  # wrap around to start
        else:
            next_line=1
        print(f"✅ Fetched successfully. Next line: {next_line}")

        return (model_name, last_modified, tags, model_card_text, next_line)

import re

class QwenResponseExtract:
    """
    🧠 Custom node to extract the final response from Qwen model output.
    It removes everything before and including </think>, returning only the user's visible text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_output": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Raw text output from Qwen model (may include <think> ... </think>)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_text",)
    FUNCTION = "extract_text"
    CATEGORY = "🧩 Qwen Utils"

    def extract_text(self, qwen_output: str):
        """
        Extracts text that comes after </think> tag.
        If no </think> tag is found, returns the original text.
        """
        # Regex to find content after </think>
        match = re.search(r"</think>\s*(.*)", qwen_output, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_text = match.group(1).strip()
        else:
            cleaned_text = qwen_output.strip()

        return (cleaned_text,)


# Register node

NODE_CLASS_MAPPINGS = {
    "HuggingFaceModelCardReader": HuggingFaceModelCardReaderNode,
    "HuggingFaceTrendingModels": HuggingFaceTrendingModelsNode,
    "HuggingFacePromptBuilder": HuggingFacePromptBuilder,
    "QwenResponseExtract": QwenResponseExtract
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuggingFaceModelCardReader": "🤗 HuggingFace Model Card Reader",
    "HuggingFaceTrendingModels": "🤗 Fetch HuggingFace Trending Models",
    "HuggingFacePromptBuilder": "🤗 HuggingFace Prompt Builder",
    "QwenResponseExtract": "🧩 Qwen Response Extract"
}



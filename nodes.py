import requests
import os
from typing import List
import os
import re
from huggingface_hub import ModelCard, model_info
import json
import os
import json
import torch
from transformers import pipeline
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

class PTagExtractor:
    """
    🧩 Extracts all text inside <p>...</p> tags and returns them as a list of strings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "html_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Input text containing multiple <p>...</p> tags"
                }),
            },
        }

    RETURN_TYPES = ("STRING_LIST",)
    RETURN_NAMES = ("p_tag_list",)
    FUNCTION = "extract"
    CATEGORY = "🧩 Qwen Utils"

    def extract(self, html_text: str):
        """
        Extracts all <p>...</p> content using regex.
        Returns list of strings.
        """

        # Finds ALL <p>...</p> content, including multiline
        matches = re.findall(r"<p>(.*?)</p>", html_text, re.DOTALL | re.IGNORECASE)

        # Clean whitespace
        cleaned = [m.strip() for m in matches if m.strip()]

        # Return as STRING_LIST type
        return (cleaned,)




class GoogleSearchFromList:
    """
    🔍 Takes a STRING_LIST of search terms and performs Google Custom Search for each.
    Returns a combined list of result URLs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query_list": ("STRING_LIST", {
                    "default": [],
                    "tooltip": "List of search phrases (from PTagExtractor)"
                }),
                "API_KEY": ("STRING", {
                    "default": "",
                    "tooltip": "Google API Key"
                }),
                "CSE_ID": ("STRING", {
                    "default": "",
                    "tooltip": "Google Custom Search Engine ID"
                }),
                "results_per_query": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Number of Google results per query"
                }),
            }
        }

    RETURN_TYPES = ("STRING_LIST",)
    RETURN_NAMES = ("all_result_links",)
    FUNCTION = "run_search"
    CATEGORY = "🧩 Web Utils"

    def google_search(self, query, api_key, cse_id, num_results):
        """Internal helper: executes Google Search API call."""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": num_results
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract result links
        results = []
        if "items" in data:
            for item in data["items"]:
                if "link" in item:
                    results.append(item["link"])

        return results

    def run_search(self, query_list, API_KEY, CSE_ID, results_per_query):
        """
        For each query in query_list:
        - Search Google
        - Append URLs to allresultslinks[]
        """

        allresultslinks = []

        for query in query_list:
            query = query.strip()
            if not query:
                continue

            try:
                google_results = self.google_search(query, API_KEY, CSE_ID, results_per_query)
                for url in google_results:
                    allresultslinks.append(url)
            except Exception as e:
                # Store error but keep pipeline running
                allresultslinks.append(f"ERROR: {query} --> {str(e)}")
        allresultslinks = list(dict.fromkeys(allresultslinks))    

        return (allresultslinks,)




class PageData:
    """Simple container for scraped result."""
    def __init__(self, link, content):
        self.link = link
        self.content = content

    def __repr__(self):
        return f"PageData(link={self.link}, content_length={len(self.content)})"


class JinaPageScraper:
    """
    🌐 Takes a list of URLs and fetches page content using Jina Reader API.
    Saves each result as numbered TXT file.
    Returns a list of PageData objects (link + page_content).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_list": ("STRING_LIST", {
                    "default": [],
                    "tooltip": "List of URLs to scrape using Jina Reader API"
                }),
                "JINA_API_KEY": ("STRING", {
                    "default": "",
                    "tooltip": "Jina Reader API Key (optional)"
                }),
                "scrape_first_n_links": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Number of Google results per query"
                }),
            }
        }

    RETURN_TYPES = ("OBJECT_LIST",)
    RETURN_NAMES = ("scraped_pages",)
    FUNCTION = "scrape"
    CATEGORY = "🧩 Web Utils"

    def fetch_with_jina(self, url: str, api_key: str):
        """
        Calls Jina Reader API:
        https://r.jina.ai/<url>
        Returns extracted clean content.
        """

        api_url = "https://r.jina.ai/" + url

        headers = {
            "x-respond-with": "markdown"
        }

        if api_key and len(api_key.strip()) > 0:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.get(api_url, headers=headers)
        resp.raise_for_status()

        return resp.text

    def scrape(self, url_list, JINA_API_KEY, scrape_first_n_links):
        """
        Loops through URLs -> fetches content -> saves TXT file -> returns list of PageData objects.
        """
        if scrape_first_n_links:
            url_list = url_list[:scrape_first_n_links]
        results = []
        counter = 1  # Start numbering files 1.txt, 2.txt ...
        # ---------------------------
        # Ensure ledger.txt exists
        # ---------------------------
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_folder, exist_ok=True)
        ledger_path = os.path.join(data_folder, "ledger.txt")
        counter_path = os.path.join(data_folder, "counter.txt")
    
        if not os.path.exists(ledger_path):
            with open(ledger_path, "w") as f:
                pass  # create empty file
        if not os.path.exists(counter_path):
            with open(counter_path, "w") as f:
                pass  # create empty file
            with open(counter_path, "w", encoding="utf-8") as f:
                        f.write("1")
        with open(ledger_path, "r") as f:
            content = f.read().strip()   # read, remove spaces/newlines
        try:
            value = int(content)+1         # convert to int
            counter=value
        except ValueError:
            value = 1  # or raise error — depends on what you want

        # Load existing links into a set
        with open(ledger_path, "r") as f:
            ledger_links = set(line.strip() for line in f if line.strip())
    
        # ---------------------------
        # Begin scraping loop
        # ---------------------------
        for link in url_list:
            link = link.strip()
            
            if not link:
                continue
            # ---------------------------
            # Skip if already processed
            # ---------------------------
            if link in ledger_links:
                print(f"[SKIP] Already in ledger: {link}")
                continue
            try:
                content = self.fetch_with_jina(link, JINA_API_KEY)

                # -------------------------
                # SAVE CONTENT AS TXT FILE
                # -------------------------
                filename = f"{counter}.txt"
                file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"LINK: {link}\n")
                    f.write("######\n")
                    f.write(content)
                with open(ledger_path, "a") as f:
                    f.write(link + "###" + filename)
                with open(counter_path, "w", encoding="utf-8") as f:
                        f.write(counter)
                counter += 1

                # Now add the object
                results.append(PageData(link, content))

            except Exception as e:
                error_text = f"ERROR: {str(e)}"

                # Save error file too
                filename = f"{counter}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"LINK: {link}\n")
                    f.write("######\n")
                    f.write(error_text)

                counter += 1

                results.append(PageData(link, error_text))

        return (results,)



class GPTOSSHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_message": ("STRING", {"multiline": True, "default": ""}),
                "your_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model_path": ("STRING", {"multiline": True, "default": ""}),  # unused but kept for compatibility
                "model_name": ("STRING", {"multiline": True, "default": "openai/gpt-oss-20b"}),
                "n_gpu_layers": ("INT", {"default": 0, "min": 0, "max": 1}),  # unused but required for compatibility
                "n_ctx": ("INT", {"default": 8192, "min": 24, "max": 20000}),
                "max_tokens": ("INT", {"default": 512, "min": 24, "max": 20000}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("gptoss_response",)
    FUNCTION = "generate_response_gptoss"
    OUTPUT_NODE = True
    CATEGORY = "utils"

    def generate_response_gptoss(
        self,
        system_message,
        your_prompt,
        model_path,
        model_name,
        n_gpu_layers,
        n_ctx,
        max_tokens,
    ):

        # Final model ID (can be HF hub ID)
        model_id = model_name.strip() or "openai/gpt-oss-20b"

        print(f"Loading GPT-OSS model: {model_id}")

        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )

        # Build GPT-OSS chat structure
        messages = []
        if system_message.strip():
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": your_prompt})

        # Run inference
        outputs = pipe(
            messages,
            max_new_tokens=max_tokens,
        )

        # Extract the final assistant message
        generated = outputs[0]["generated_text"][-1]

        # Save raw result to a JSON file (same behaviour as your Qwen node)
        caption_file_path = os.path.join(model_path, "gptoss_response.json")
        try:
            with open(caption_file_path, "w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=2)
        except:
            print("[WARN] Cannot save output JSON. Check write permissions.")

        return (generated,)

# Register node

NODE_CLASS_MAPPINGS = {
    "HuggingFaceModelCardReader": HuggingFaceModelCardReaderNode,
    "HuggingFaceTrendingModels": HuggingFaceTrendingModelsNode,
    "HuggingFacePromptBuilder": HuggingFacePromptBuilder,
    "QwenResponseExtract": QwenResponseExtract,
    "PTagExtractor": PTagExtractor,
    "GoogleSearchFromList": GoogleSearchFromList,
    "JinaPageScraper": JinaPageScraper,
    "GPTOSSHelper": GPTOSSHelper,
    
    
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuggingFaceModelCardReader": "🤗 HuggingFace Model Card Reader",
    "HuggingFaceTrendingModels": "🤗 Fetch HuggingFace Trending Models",
    "HuggingFacePromptBuilder": "🤗 HuggingFace Prompt Builder",
    "QwenResponseExtract": "🧩 Qwen Response Extract",
    "PTagExtractor": "🧩 P Tag Extractor",
    "GoogleSearchFromList": "🧩 Google Search From List",
    "JinaPageScraper": "🧩 Jina Page Scraper",
    "GPTOSSHelper": "GPT OSS Helper"
    
}



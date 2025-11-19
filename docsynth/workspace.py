
import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

# NOTE: we import google.generativeai as genai just like your original code.
# Depending on the genai SDK version you have, the exact call to generate
# may differ. We call `self._generate_with_gemini(prompt, image)` which
# encapsulates the SDK call so you can adapt it if your installed SDK uses
# different method names.
try:
    import google.generativeai as genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    _HAS_GENAI = False

logger = logging.getLogger("DocumentClassifier")
logger.setLevel(logging.INFO)

# Reasonable default limits (tunable)
MAX_IMAGE_BYTES = 6 * 1024 * 1024  # 6 MB decoded image
ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/tiff"}


@dataclass
class ClassificationRule:
    document_class: str
    conditions: List[str]
    description: str = ""


class DocumentClassifier:
    """
    Document classifier using Gemini (google.generativeai).
    - Keeps an in-memory list of rules (ClassificationRule)
    - add_rule is idempotent by document_class
    - provides remove_rule, clear_rules, list_rules
    - robust image/base64 handling and safe JSON extraction from model responses

    NOTE: If you want persistence of rules across restarts, persist the output
    of list_rules() to disk or to your server's rule store (server.py does that).
    """

    def __init__(self, api_key: str = "", model: str = "gemini-1.5-flash", mock_mode: bool = False):
        """
        Args:
            api_key: Google API key for Gemini (optional if mock_mode=True)
            model: model identifier (kept as a string)
            mock_mode: If True, classifier will not call the Gemini API and
                       will return deterministic mock responses for testing.
        """
        self.model_name = model
        self.api_key = api_key
        self.mock_mode = mock_mode

        # Configure genai if available and not mock mode
        if self.mock_mode:
            logger.info("DocumentClassifier running in mock_mode (no external calls).")
        else:
            if not _HAS_GENAI:
                logger.warning("google.generativeai package not available. Set mock_mode=True for local testing.")
            else:
                # Avoid logging the API key itself
                genai.configure(api_key=api_key)

                # The original code used: self.model = genai.GenerativeModel(model)
                # Different genai versions have different usage patterns; we keep
                # `self.model_client` as a thin wrapper. We'll still store model_name.
                try:
                    # If SDK exposes a GenerativeModel class as in your code:
                    self.model_client = genai.GenerativeModel(model)  # type: ignore
                except Exception:
                    # Fallback: some genai SDKs use genai.get_model(...) or genai.create(...)
                    # We'll keep model_client None and handle it in _generate_with_gemini
                    logger.info("Could not construct GenerativeModel via SDK; will call genai.generate if available.")
                    self.model_client = None  # type: ignore

        # configurable generation config
        self.generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        # rules storage
        self.classification_rules: List[ClassificationRule] = []

    #
    # Rule management (idempotent where it makes sense)
    #
    def add_rule(self, document_class: str, conditions: List[str], description: str = "") -> None:
        """
        Add or update a rule. If a rule for document_class already exists, it is replaced.
        """
        # Validate inputs minimally
        if not document_class or not isinstance(conditions, list):
            raise ValueError("document_class (str) and conditions (List[str]) required")

        # Replace if exists
        for r in self.classification_rules:
            if r.document_class == document_class:
                r.conditions = conditions
                r.description = description
                logger.info("Updated existing rule for class: %s", document_class)
                return

        # Otherwise append
        self.classification_rules.append(ClassificationRule(document_class=document_class,
                                                            conditions=conditions,
                                                            description=description))
        logger.info("Added rule for class: %s", document_class)

    def remove_rule(self, document_class: str) -> bool:
        """
        Remove a rule by document_class. Returns True if removed, False if not found.
        """
        for i, r in enumerate(self.classification_rules):
            if r.document_class == document_class:
                self.classification_rules.pop(i)
                logger.info("Removed rule for class: %s", document_class)
                return True
        logger.debug("Attempted to remove missing rule: %s", document_class)
        return False

    def clear_rules(self) -> None:
        """Remove all rules."""
        self.classification_rules.clear()
        logger.info("Cleared all classification rules.")

    def list_rules(self) -> List[Dict]:
        """Return rule list as serializable dicts."""
        return [
            {"document_class": r.document_class, "conditions": r.conditions, "description": r.description}
            for r in self.classification_rules
        ]

    #
    # Internal helpers
    #
    def _is_base64(self, s: str) -> bool:
        try:
            base64.b64decode(s, validate=True)
            return True
        except Exception:
            return False

    def _prepare_image_from_base64(self, document_base64: str, media_type: str):
        """
        Returns a PIL Image object after decoding base64. Raises ValueError for bad input.
        """
        if not isinstance(document_base64, str) or not document_base64:
            raise ValueError("document_base64 must be a non-empty string")

        if not self._is_base64(document_base64):
            raise ValueError("document_base64 does not appear to be valid base64")

        # Estimate decoded size and protect against huge uploads
        estimated_decoded_bytes = (len(document_base64) * 3) // 4
        if estimated_decoded_bytes > MAX_IMAGE_BYTES:
            raise ValueError("Decoded image exceeds maximum allowed size")

        # Basic media-type check
        if media_type not in ALLOWED_MEDIA_TYPES:
            raise ValueError(f"Unsupported media_type: {media_type}")

        # decode bytes and open image via PIL
        try:
            image_bytes = base64.b64decode(document_base64)
            image = Image.open(io.BytesIO(image_bytes))
            # Optionally convert to RGB to normalize formats
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except UnidentifiedImageError as e:
            raise ValueError("Could not decode image - unsupported or corrupted image data") from e

    def _build_classification_prompt(self, use_rules: bool = True) -> str:
        """
        Build the instruction prompt. Keeps your rule-based JSON-only output requirement.
        """
        if use_rules and self.classification_rules:
            rules_text = "\n\n".join(
                f"Rule {idx + 1}: {rule.document_class}\n"
                f"Identify this document type if it contains: {', '.join(rule.conditions)}\n"
                f"Description: {rule.description}"
                for idx, rule in enumerate(self.classification_rules)
            )

            prompt = f"""You are an expert document classifier.
Analyze the provided document image and classify it according to the following rules:

{rules_text}

Return ONLY a single JSON object following this exact shape:
{{ "document_class": "...", "confidence": "...", "reasoning": "...", "identified_features": [...], "alternative_classification": "...", "raw_observations": "..." }}
Do not include any explanatory text, only the JSON object."""
        else:
            prompt = """You are an expert document classifier. Analyze the provided document image and return ONLY a single JSON object with these keys:
{ "document_class": "...", "confidence": "...", "reasoning": "...", "identified_features": [...], "document_details": "...", "extracted_key_info": "..." }
Do not include any explanatory text, only the JSON."""

        return prompt

    def _clean_model_text_to_json(self, text: str) -> str:
        """
        Extract JSON substring from a model text response robustly.
        Handles code fences and stray text around JSON.
        """
        if not text:
            raise ValueError("Empty response from model")

        t = text.strip()

        # If there are code fences with json
        # Prefer ```json ... ``` then ``` ... ```
        for fence in ("```json", "```"):
            if fence in t:
                try:
                    start = t.index(fence) + len(fence)
                    # find closing ```
                    end = t.index("```", start)
                    candidate = t[start:end].strip()
                    return candidate
                except ValueError:
                    # not well-formed; fall back to other strategies
                    pass

        # Fallback: try to find first { ... } block
        first_brace = t.find("{")
        last_brace = t.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return t[first_brace:last_brace + 1]

        # give up — return original text so caller can include raw response for debugging
        return t

    def _generate_with_gemini(self, prompt: str, image) -> str:
        """
        Encapsulate calls to google.generativeai SDK for easier adaptation.
        This function should return the model's textual response.
        If the SDK in your environment exposes different functions, modify this method.
        """
        if self.mock_mode:
            # deterministic mock: return a plausible JSON string
            mock_json = {
                "document_class": "Mock Document",
                "confidence": 90,
                "reasoning": "mock_mode - no external call",
                "identified_features": ["mock_feature"],
                "document_details": "mock details"
            }
            return json.dumps(mock_json)

        if not _HAS_GENAI:
            raise RuntimeError("google.generativeai is not available in this environment")

        # Try to use `self.model_client.generate_content` if available (your original approach)
        try:
            # Some SDKs accept prompt + image in a list like your original code.
            # We try the same call but wrapped to catch failures and provide a
            # helpful error message if the SDK differs.
            response = None
            if getattr(self, "model_client", None) is not None:
                # SDK style: model_client.generate_content([prompt, image], generation_config=...)
                try:
                    response = self.model_client.generate_content([prompt, image], generation_config=self.generation_config)  # type: ignore
                    # Some SDK response objects expose .text
                    if hasattr(response, "text"):
                        return response.text
                    # Some expose content as response["candidates"][0]["content"] or similar
                    if isinstance(response, dict):
                        # try a few common shapes
                        if "candidates" in response and len(response["candidates"]) > 0:
                            return response["candidates"][0].get("content", "")
                        if "output" in response:
                            return str(response["output"])
                    # fallback to stringifying
                    return str(response)
                except Exception as inner_e:
                    logger.debug("model_client.generate_content failed: %s", inner_e)
                    # fallthrough to try another style of call

            # Fallback call: some genai versions use genai.generate_text(...) or genai.generate(...)
            try:
                # Example fallback: genai.generate(...) or genai.generate_text(...)
                if hasattr(genai, "generate"):
                    response = genai.generate(model=self.model_name, prompt=prompt, image=image, **self.generation_config)  # type: ignore
                    # attempt common extraction patterns
                    if hasattr(response, "text"):
                        return response.text
                    if isinstance(response, dict) and "output" in response:
                        return str(response["output"])
                    return str(response)
                elif hasattr(genai, "generate_text"):
                    resp = genai.generate_text(model=self.model_name, prompt=prompt, **self.generation_config)  # type: ignore
                    if hasattr(resp, "text"):
                        return resp.text
                    return str(resp)
                else:
                    raise RuntimeError("Unsupported genai SDK shape; adapt _generate_with_gemini() to your SDK.")
            except Exception as e:
                # bubble up helpful message
                raise RuntimeError(f"Failed to call Gemini via SDK. Update _generate_with_gemini for your genai version. underlying error: {e}") from e

        except Exception as e:
            raise

    #
    # Public classifier methods
    #
    def classify_document(self, document_base64: str, media_type: str = "image/jpeg", use_rules: bool = True) -> Dict:
        """
        Main classification entrypoint.
        Returns a dict with at least "success" (bool) and either the parsed fields
        from the model or helpful error information.
        """
        try:
            # Validate and prepare image
            image = self._prepare_image_from_base64(document_base64, media_type)

            # Build prompt
            prompt = self._build_classification_prompt(use_rules)

            # Get model output (text)
            model_text = self._generate_with_gemini(prompt, image)

            # Clean / extract JSON blob
            json_text = self._clean_model_text_to_json(model_text)

            # Parse JSON
            parsed = json.loads(json_text)

            # Normalize/add metadata
            if not isinstance(parsed, dict):
                raise ValueError("Parsed JSON is not an object")

            parsed.setdefault("success", True)
            parsed.setdefault("model_used", self.model_name)
            parsed.setdefault("rules_available", len(self.classification_rules))

            return parsed

        except json.JSONDecodeError as jde:
            logger.exception("Failed to decode JSON from model response")
            return {
                "success": False,
                "error": "Failed to parse JSON from model",
                "parse_error": str(jde),
                "raw_response": locals().get("model_text", "")
            }
        except Exception as e:
            logger.exception("Classification failed")
            return {
                "success": False,
                "error": str(e),
                "document_class": "Error",
                "confidence": "none",
                "reasoning": f"Classification failed: {str(e)}"
            }

    def classify_multiple_documents(self, documents: List[Tuple[str, str]], use_rules: bool = True) -> List[Dict]:
        results = []
        for idx, (doc_base64, media_type) in enumerate(documents):
            logger.info("Classifying document %d/%d", idx + 1, len(documents))
            r = self.classify_document(doc_base64, media_type, use_rules)
            r["document_index"] = idx
            results.append(r)
        return results

    def classify_from_file(self, file_path: str, use_rules: bool = True) -> Dict:
        # read file bytes and base64
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        doc_base64 = base64.b64encode(file_bytes).decode("utf-8")

        # pick media type from extension (simple heuristic)
        lower = file_path.lower()
        if lower.endswith(".png"):
            media_type = "image/png"
        elif lower.endswith(".jpg") or lower.endswith(".jpeg"):
            media_type = "image/jpeg"
        elif lower.endswith(".tif") or lower.endswith(".tiff"):
            media_type = "image/tiff"
        elif lower.endswith(".pdf"):
            # We don't handle multipage PDFs here; user should convert PDF pages to images
            raise ValueError("PDF support not implemented. Convert PDF page(s) to image(s) first.")
        else:
            media_type = "image/jpeg"

        return self.classify_document(doc_base64, media_type, use_rules)

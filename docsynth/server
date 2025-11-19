# server.py
from fastmcp import FastMCP
from typing import List, Dict, Any, Optional
import json
import base64
import os
import threading
import logging
from dataclasses import asdict
from pathlib import Path

# Import your DocumentClassifier and ClassificationRule from your existing code
# EXPECTED DocumentClassifier interface:
# class DocumentClassifier:
#     def __init__(self, api_key: str, model: str): ...
#     def add_rule(self, document_class: str, conditions: List[str], description: str = "") -> None: ...
#     def remove_rule(self, document_class: str) -> None: ...
#     def list_rules(self) -> List[Dict[str, Any]]: ...  # optional but useful
#     def classify_document(self, document_base64: str, media_type: str = "image/jpeg", use_rules: bool = True) -> Dict[str, Any]
from workspace import DocumentClassifier

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docmcp")

# Configuration
RULES_PERSISTENCE_FILE = os.environ.get("DOCSYNTH_RULES_FILE", "classifier_rules.json")
MAX_BASE64_BYTES = int(os.environ.get("DOCSYNTH_MAX_BASE64_BYTES", 4 * 1024 * 1024))  # 4MB default

mcp = FastMCP(
    name="DocSynthAI MCP Server",
    version="1.0.0"
)

# Global state with lock for thread safety
classifier_state = {
    "classifier": None,         # DocumentClassifier instance
    "api_key": None,            # stored only to re-init; avoid printing
    "model": "gemini-1.5-flash",# consistent default
    "rules": []                 # list of dicts: {"document_class":..., "conditions":..., "description":...}
}
_state_lock = threading.RLock()

# Helpers for persistence
def _load_rules_from_disk():
    path = Path(RULES_PERSISTENCE_FILE)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            rules = json.load(f)
            if isinstance(rules, list):
                return rules
            logger.warning("Rules file has unexpected format, ignoring.")
            return []
    except Exception as e:
        logger.exception("Failed loading rules file, starting with empty rules.")
        return []

def _save_rules_to_disk(rules: List[Dict[str, Any]]):
    try:
        with open(RULES_PERSISTENCE_FILE, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Failed to persist rules to disk.")

# Initialize in-memory rules from disk on startup
with _state_lock:
    classifier_state["rules"] = _load_rules_from_disk()

def initialize_classifier(api_key: str, model: Optional[str] = None) -> DocumentClassifier:
    """
    Create or recreate a DocumentClassifier and restore rules.

    Notes:
    - Avoid logging or printing api_key.
    - This will call DocumentClassifier.add_rule for each saved rule.
    - DocumentClassifier.add_rule SHOULD be idempotent (or you should guard duplicates here).
    """
    if model is None:
        model = classifier_state.get("model", "gemini-1.5-flash")

    logger.info("Initializing DocumentClassifier with model: %s", model)
    # instantiate classifier
    classifier = DocumentClassifier(api_key=api_key, model=model)

    # restore rules
    # If your DocumentClassifier has a clear_rules() method, call it first to avoid duplicates.
    for rule_dict in classifier_state["rules"]:
        try:
            classifier.add_rule(
                document_class=rule_dict["document_class"],
                conditions=rule_dict["conditions"],
                description=rule_dict.get("description", "")
            )
        except Exception:
            # do not propagate rule failures; log and continue
            logger.exception("Failed to add rule %s while initializing classifier", rule_dict.get("document_class"))

    with _state_lock:
        classifier_state["classifier"] = classifier
        classifier_state["api_key"] = api_key
        classifier_state["model"] = model

    return classifier

@mcp.tool()
def setup_classifier(api_key: str, model: str = "gemini-1.5-flash") -> Dict[str, Any]:
    """
    Initialize the document classifier with your API key.
    """
    try:
        initialize_classifier(api_key=api_key, model=model)
        return {
            "success": True,
            "message": f"Classifier initialized successfully with model: {model}",
            "model": model,
            "rules_count": len(classifier_state["rules"])
        }
    except Exception as e:
        logger.exception("Failed to initialize classifier")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to initialize classifier"
        }

@mcp.tool()
def create_rule(document_class: str, conditions: List[str], description: str = "") -> Dict[str, Any]:
    """
    Create or update a rule. Rules persisted to disk.
    """
    with _state_lock:
        if classifier_state["classifier"] is None:
            return {"success": False, "error": "Classifier not initialized", "message": "Call setup_classifier first"}

        # normalize rule structure
        rule_dict = {
            "document_class": document_class,
            "conditions": conditions,
            "description": description
        }

        # find existing
        existing_index = next((i for i, r in enumerate(classifier_state["rules"])
                               if r.get("document_class") == document_class), None)
        try:
            classifier_state["classifier"].add_rule(
                document_class=document_class,
                conditions=conditions,
                description=description
            )
        except Exception:
            logger.exception("Failed to add/update rule in classifier instance")
            return {"success": False, "error": "Failed to add rule to classifier instance"}

        if existing_index is not None:
            classifier_state["rules"][existing_index] = rule_dict
            action = "updated"
        else:
            classifier_state["rules"].append(rule_dict)
            action = "created"

        # persist
        _save_rules_to_disk(classifier_state["rules"])

        return {
            "success": True,
            "message": f"Rule {action} successfully",
            "action": action,
            "rule": rule_dict,
            "total_rules": len(classifier_state["rules"])
        }

@mcp.tool()
def get_all_rules() -> Dict[str, Any]:
    with _state_lock:
        return {
            "success": True,
            "rules": list(classifier_state["rules"]),
            "total_rules": len(classifier_state["rules"]),
            "document_classes": [r["document_class"] for r in classifier_state["rules"]],
            "classifier_initialized": classifier_state["classifier"] is not None
        }

@mcp.tool()
def delete_rule(document_class: str) -> Dict[str, Any]:
    with _state_lock:
        rule_index = next((i for i, r in enumerate(classifier_state["rules"])
                           if r["document_class"] == document_class), None)
        if rule_index is None:
            return {
                "success": False,
                "error": f"No rule found for document class: {document_class}",
                "available_classes": [r["document_class"] for r in classifier_state["rules"]]
            }

        deleted_rule = classifier_state["rules"].pop(rule_index)
        # persist changes
        _save_rules_to_disk(classifier_state["rules"])

        # Reinitialize classifier if needed to remove the rule from the model side
        try:
            if classifier_state.get("api_key"):
                initialize_classifier(api_key=classifier_state["api_key"], model=classifier_state["model"])
        except Exception:
            logger.exception("Failed to reinitialize classifier after deleting rule")

        return {
            "success": True,
            "message": f"Rule for '{document_class}' deleted successfully",
            "deleted_rule": deleted_rule,
            "remaining_rules": len(classifier_state["rules"])
        }

# Helper: validate base64-ish string (fast check)
def _is_base64(s: str) -> bool:
    try:
        # Python base64.b64decode will raise if not properly padded or invalid
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

@mcp.tool()
def classify_document(document_base64: str, media_type: str = "image/jpeg", use_rules: bool = True) -> Dict[str, Any]:
    """
    Classify a base64-encoded image.

    - Validates input
    - Enforces size limits
    - Delegates to DocumentClassifier.classify_document
    """
    with _state_lock:
        if classifier_state["classifier"] is None:
            return {
                "success": False,
                "error": "Classifier not initialized",
                "message": "Call setup_classifier first to initialize the system",
                "document_class": "Error",
                "confidence": 0
            }

    # Basic media-type whitelist
    allowed_media_types = {"image/jpeg", "image/png", "image/tiff"}
    if media_type not in allowed_media_types:
        return {"success": False, "error": f"Unsupported media_type: {media_type}", "confidence": 0}

    if not document_base64 or not isinstance(document_base64, str):
        return {"success": False, "error": "Invalid document_base64", "confidence": 0}

    # Quick base64 validation and size cap
    if not _is_base64(document_base64):
        return {"success": False, "error": "document_base64 does not appear to be valid base64", "confidence": 0}

    # Heuristic: estimate decoded size: each 4 chars -> 3 bytes
    estimated_decoded_bytes = (len(document_base64) * 3) // 4
    if estimated_decoded_bytes > MAX_BASE64_BYTES:
        return {"success": False, "error": "Document exceeds maximum allowed size", "confidence": 0}

    try:
        with _state_lock:
            result = classifier_state["classifier"].classify_document(
                document_base64=document_base64,
                media_type=media_type,
                use_rules=use_rules
            )

        # Canonicalize expected fields and add metadata
        if not isinstance(result, dict):
            logger.warning("Classifier returned non-dict result")
            result = {"success": False, "error": "Invalid classifier response", "confidence": 0}

        result.setdefault("classification_mode", "rule-based" if use_rules else "general")
        result.setdefault("rules_available", len(classifier_state["rules"]))
        result.setdefault("model_used", classifier_state.get("model"))

        return result

    except Exception as e:
        logger.exception("Classification failed")
        return {
            "success": False,
            "error": str(e),
            "document_class": "Error",
            "confidence": 0,
            "reasoning": f"Classification failed: {str(e)}"
        }

if __name__ == "__main__":
    # Configurable transport via environment variables
    TRY_HTTP = os.environ.get("DOCSYNTH_TRY_HTTP", "1") == "1"
    if TRY_HTTP:
        host = os.environ.get("DOCSYNTH_HOST", "0.0.0.0")
        port = int(os.environ.get("DOCSYNTH_PORT", "8000"))
        logger.info("Starting MCP server with HTTP transport on %s:%d", host, port)
        mcp.run(transport="http", host=host, port=port)
    else:
        logger.info("Starting MCP server with stdio transport")
        mcp.run(transport="stdio")

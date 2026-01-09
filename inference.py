"""
Inference wrapper: loads model and provides predict(text) -> dict
with score, label, evidence, and recommended action.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "model_out"

# Fallback labels if no model exists
DEFAULT_LABELS = ["benign", "phishing", "fraud"]


def _load_model():
    if os.path.isdir(MODEL_DIR):
        print(f"Loading fine-tuned model from {MODEL_DIR} ...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        return model, tokenizer
    else:
        print("⚠️ No trained model found. Using heuristic fallback classifier.")
        return None, None


model, tokenizer = _load_model()


def heuristic_classifier(text):
    """Very simple fallback rule-based detection."""
    text_l = text.lower()

    if any(keyword in text_l for keyword in ["bank", "verify", "password", "account", "login", "click"]):
        return {
            "label": "phishing",
            "score": 0.82,
            "action": "Do NOT click links. Report and delete."
        }

    if any(keyword in text_l for keyword in ["lottery", "prize", "winner", "claim now", "urgent"]):
        return {
            "label": "fraud",
            "score": 0.76,
            "action": "Do not reply. Block sender."
        }

    return {
        "label": "benign",
        "score": 0.10,
        "action": "Safe message."
    }


def predict(text: str):
    """Runs prediction using model if available, otherwise heuristics."""

    if model is None:
        return heuristic_classifier(text)

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

    score, idx = torch.max(probs, dim=0)
    label = model.config.id2label[idx.item()]

    # Recommended actions
    actions = {
        "benign": "Safe message.",
        "phishing": "Do NOT click links. Report immediately.",
        "fraud": "Block sender and report."
    }

    return {
        "label": label,
        "score": float(score),
        "action": actions.get(label, "Proceed with caution.")
    }


if __name__ == "__main__":
    print("Real-Time Threat Detector")
    print("Type a message below.\n")

    while True:
        text = input("Enter text (or type exit): ")
        if text.lower() in ("exit", "quit"):
            break

        result = predict(text)
        print("\nPrediction:", result, "\n")

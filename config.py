"""
Configuration for the AI Science Discovery Team.
Maps each agent role to its LM Studio model and parameters.

IMPORTANT: Since LM Studio typically loads ONE model at a time,
the system will auto-detect whichever model is currently loaded.
Just load your best model (GPT-OSS 20B recommended) and the system
will use it for all agents, with different temperatures and prompts
to create distinct "personalities" for each role.
"""

import requests

# ═══════════════════════════════════════════════════════════════════════
# LM STUDIO API CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"  # LM Studio doesn't require a real key

# ═══════════════════════════════════════════════════════════════════════
# MODEL DETECTION
# Auto-detect the loaded model, or fall back to a default name.
# LM Studio will use whatever model is loaded regardless of the name
# sent in the API request, but we try to match for cleanliness.
# ═══════════════════════════════════════════════════════════════════════

def detect_loaded_model():
    """Auto-detect which model is currently loaded in LM Studio."""
    try:
        resp = requests.get(
            f"{LM_STUDIO_BASE_URL}/models",
            headers={"Authorization": f"Bearer {LM_STUDIO_API_KEY}"},
            timeout=5,
        )
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                return models[0].get("id", "auto-detect")
    except Exception:
        pass
    return "auto-detect"

# Try to detect at import time; will re-detect at runtime if needed
_DETECTED_MODEL = detect_loaded_model()

# Set to True to auto-detect and use whatever single model is loaded.
# Set to False to specify exact model IDs (requires multi-model loading).
USE_AUTO_DETECT = False

if USE_AUTO_DETECT:
    MODELS = {
        "gpt_oss_20b": _DETECTED_MODEL,
        "llama_31_8b": _DETECTED_MODEL,
        "falcon_h1_7b": _DETECTED_MODEL,
        "mistral_7b": _DETECTED_MODEL,
        "phi3_mini": _DETECTED_MODEL,
    }
else:
    # Exact model IDs provided by user
    MODELS = {
        "gpt_oss_20b": "gpt-oss-20b",
        "llama_31_8b": "meta-llama-3.1-8b-instruct",
        "falcon_h1_7b": "falcon-h1-7b-instruct",
        "mistral_7b": "mistral-7b-instruct-v0.3",
        "phi3_mini": "phi-3-mini-4k-instruct",
    }

# ═══════════════════════════════════════════════════════════════════════
# AGENT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════

AGENT_CONFIGS = {
    # ─────────────────────────────────────────────────────────────────
    # STEP 1-2: ORCHESTRATOR (Complex Reasoning -> GPT-OSS)
    # ─────────────────────────────────────────────────────────────────
    "orchestrator": {
        "model": MODELS["gpt_oss_20b"],
        "temperature": 0.3,
        "max_tokens": 2000,
        "top_p": 0.9,
    },

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: HYPOTHESIS GENERATOR (Creative Reasoning -> GPT-OSS)
    # ─────────────────────────────────────────────────────────────────
    "hypothesis_generator": {
        "model": MODELS["gpt_oss_20b"],
        "temperature": 0.8,
        "max_tokens": 4000,
        "top_p": 0.95,
    },

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: STEP DECOMPOSER (Structure/Formatting -> Llama 3.1)
    # ─────────────────────────────────────────────────────────────────
    "step_decomposer": {
        "model": MODELS["llama_31_8b"],
        "temperature": 0.2,
        "max_tokens": 3000,
        "top_p": 0.9,
    },

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: PHYSICS ORACLE (Deep Reasoning -> GPT-OSS)
    # ─────────────────────────────────────────────────────────────────
    "physics_oracle": {
        "model": MODELS["gpt_oss_20b"],
        "temperature": 0.2,
        "max_tokens": 2000,
        "top_p": 0.85,
    },

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: CHAIN ASSEMBLER (Structure/Formatting -> Llama 3.1)
    # ─────────────────────────────────────────────────────────────────
    "chain_assembler": {
        "model": MODELS["llama_31_8b"],
        "temperature": 0.3,
        "max_tokens": 3000,
        "top_p": 0.9,
    },

    # ─────────────────────────────────────────────────────────────────
    # STEP 7: ENGINEERING PROPOSER (Creative Reasoning -> GPT-OSS)
    # ─────────────────────────────────────────────────────────────────
    "engineering_proposer": {
        "model": MODELS["gpt_oss_20b"],
        "temperature": 0.7,
        "max_tokens": 4000,
        "top_p": 0.95,
    },

    # ─────────────────────────────────────────────────────────────────
    # STEP 8: REQUIREMENT CHALLENGER (Critical Thinking -> GPT-OSS)
    # ─────────────────────────────────────────────────────────────────
    "requirement_challenger": {
        "model": MODELS["gpt_oss_20b"],
        "temperature": 0.9,
        "max_tokens": 3000,
        "top_p": 0.95,
    },

    # ─────────────────────────────────────────────────────────────────
    # STEP 9: OVERSEER / SYNTHESIZER (Complex Context -> GPT-OSS)
    # ─────────────────────────────────────────────────────────────────
    "overseer": {
        "model": MODELS["gpt_oss_20b"],
        "temperature": 0.4,
        "max_tokens": 4000,
        "top_p": 0.9,
    },

    # ─────────────────────────────────────────────────────────────────
    # STEP 10: FINAL EVALUATOR (Writing -> GPT-OSS)
    # ─────────────────────────────────────────────────────────────────
    "final_evaluator": {
        "model": MODELS["gpt_oss_20b"],
        "temperature": 0.3,
        "max_tokens": 6000,
        "top_p": 0.9,
    },
}

# ═══════════════════════════════════════════════════════════════════════
# PIPELINE SETTINGS
# ═══════════════════════════════════════════════════════════════════════

# How many approaches the Hypothesis Generator should propose (Step 3)
NUM_HYPOTHESES = 5

# How many times the Requirement Challenger loops (Step 8)
CHALLENGE_ITERATIONS = 3

# How many top proposals the Final Evaluator should write up (Step 10)
NUM_FINAL_PROPOSALS = 3

# Results output directory
RESULTS_DIR = "results"

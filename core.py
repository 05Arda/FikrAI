from datetime import datetime, UTC

MODEL_ID = "unsloth/gemma-4-E2B-it"
OUTPUT_DIR = "finetuned_models" / f"./model_{datetime.now(UTC).timestamp()}"
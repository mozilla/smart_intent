# This module help sload the intent model from checkpoint specified
# Uploads to the mentioned HF space

import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from constants import (INTENT_MODEL_CHECKPOINT,
                       INTENT_ID2LABEL,
                       INTENT_LABEL2ID,
                       NUM_LABELS,
                       INTENT_MODEL_OUTPUT_DIR,
                       MODEL_ARTIFACT_DIR,
                       INTENT_MODEL_REPO_ID)

class IntentModelArtifact:
    """ To run CHKPT_DIR=<checkpoint-#####> python src/upload_intent.py"""
    base_model = AutoModelForSequenceClassification.from_pretrained(
        INTENT_MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
        id2label=INTENT_ID2LABEL,
        label2id=INTENT_LABEL2ID)
    
    def __init__(self):
        CHKPT_DIR = os.getenv("CHKPT_DIR", "NO_DIR")
        output_dir = f"{INTENT_MODEL_OUTPUT_DIR}/{CHKPT_DIR}"
        if not os.path.exists(output_dir):
            raise ValueError(f"{output_dir} does not exist. Can you check your CHKPT_DIR env variable ")
        lora_model = PeftModel.from_pretrained(self.base_model, output_dir)
        self.merged_model = lora_model.merge_and_unload()
        self.merged_model.save_pretrained(MODEL_ARTIFACT_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)  # Load the tokenizer
        self.tokenizer.save_pretrained(MODEL_ARTIFACT_DIR)

    def upload_model_artifacts(self):
        self.merged_model.push_to_hub(INTENT_MODEL_REPO_ID)
        self.tokenizer.push_to_hub(INTENT_MODEL_REPO_ID)


if __name__ == '__main__':
    intent_model_artifact = IntentModelArtifact()
    intent_model_artifact.upload_model_artifacts()

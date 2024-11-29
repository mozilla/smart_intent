# This module help sload the intent model from checkpoint specified
# Uploads to the mentioned HF space

import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
from peft import PeftModel
from constants import (NER_MODEL_CHECKPOINT,
                       NER_ID2LABEL,
                       NER_LABEL2ID,
                       NER_NUM_LABELS,
                       NER_MODEL_OUTPUT_DIR,
                       NER_MODEL_ARTIFACT_DIR,
                       NER_MODEL_REPO_ID)

class NERModelArtifact:
    """ To run CHKPT_DIR=<checkpoint-#####> python src/upload_intent.py"""
    base_model = AutoModelForTokenClassification.from_pretrained(
        NER_MODEL_CHECKPOINT,
        num_labels=NER_NUM_LABELS,
        id2label=NER_ID2LABEL,
        label2id=NER_LABEL2ID)
    
    def __init__(self):
        CHKPT_DIR = os.getenv("CHKPT_DIR", "NO_DIR")
        output_dir = f"{NER_MODEL_OUTPUT_DIR}/{CHKPT_DIR}"
        if not os.path.exists(output_dir):
            raise ValueError(f"{output_dir} does not exist. Can you check your CHKPT_DIR env variable ")
        lora_model = PeftModel.from_pretrained(self.base_model, output_dir)
        self.merged_model = lora_model.merge_and_unload()
        self.merged_model.save_pretrained(NER_MODEL_ARTIFACT_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)  # Load the tokenizer
        self.tokenizer.save_pretrained(NER_MODEL_ARTIFACT_DIR)

    def upload_model_artifacts(self):
        self.merged_model.push_to_hub(NER_MODEL_REPO_ID)
        self.tokenizer.push_to_hub(NER_MODEL_REPO_ID)


if __name__ == '__main__':
    ner_model_artifact = NERModelArtifact()
    ner_model_artifact.upload_model_artifacts()

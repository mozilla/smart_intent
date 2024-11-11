# This module helps train the Intent Classifier by finetuning Mobilebert using LoRA

from datasets import load_dataset
from transformers import (AutoTokenizer,
                         AutoModelForSequenceClassification,
                         DataCollatorWithPadding,
                         TrainingArguments,
                         Trainer,
                         EarlyStoppingCallback)

from peft import get_peft_model, LoraConfig
from constants import (INTENT_LABEL2ID,
                       INTENT_ID2LABEL,
                       NUM_LABELS,
                       INTENT_DATASET_REPO_ID,
                       INTENT_MODEL_CHECKPOINT,
                       INTENT_MODEL_OUTPUT_DIR)
from utils import compute_metrics
from torch import cuda
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntentClassificationTrainer")
device = 'cuda' if cuda.is_available() else 'mps'
logger.info(f"Using device: {device}")

class IntentClassificationTrainer:
    peft_config = LoraConfig(task_type="SEQ_CLS",
                             r=8, # intrinsic rank of trainable weight matrix
                             lora_alpha=32, # similar to learning_rate
                             lora_dropout=0.01, # probability of dropout nodes
                             target_modules=['attention.self.query']) # LoRA is applied to query layer
    lr = 1e-4
    batch_size = 32
    num_epochs = 12
    weight_decay = 0.01

    # training args
    training_args = TrainingArguments(
        output_dir=INTENT_MODEL_OUTPUT_DIR,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            INTENT_MODEL_CHECKPOINT,
            num_labels=NUM_LABELS,
            id2label=INTENT_ID2LABEL,
            label2id=INTENT_LABEL2ID
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            INTENT_MODEL_CHECKPOINT,
            add_prefix_space=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = get_peft_model(self.model, self.peft_config).to(device)
        self.model.print_trainable_parameters()


    def _load_data(self):
        return load_dataset(INTENT_DATASET_REPO_ID)
    
    def _tokenize_function(self, examples):
        # extract text
        text = examples["sequence"]

        # tokenize and truncate text
        self.tokenizer.truncation_side = "right"
        tokenized_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,  # Pad the sequences to the longest in the batch
            max_length=64
        )
        return tokenized_inputs

    def preprocess_data(self, dataset_dict):
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return dataset_dict.map(self._tokenize_function, batched=True)
    
    def run(self, tokenized_dataset):
        trainer = Trainer(
            model=self.model,
            args=self.training_args, # Hyperparamaters
            train_dataset=tokenized_dataset["train"], # training data
            eval_dataset=tokenized_dataset["validation"], # validation data
            tokenizer=self.tokenizer, # tokenizer
            data_collator=self.data_collator, # dynamic sequence padding
            compute_metrics=compute_metrics,  # model perfomance evaluation metric
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()


if __name__ == '__main__':
    intentTrainer = IntentClassificationTrainer()
    dataset_dict = intentTrainer._load_data()
    tokenized_dataset = intentTrainer.preprocess_data(dataset_dict)
    intentTrainer.run(tokenized_dataset)

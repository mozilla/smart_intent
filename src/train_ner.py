# This module helps train the NER by finetuning Distilbert using LoRA

from datasets import load_dataset
from transformers import (AutoTokenizer,
                         AutoModelForTokenClassification,
                         TrainingArguments,
                         Trainer)

from peft import get_peft_model, LoraConfig, TaskType
from constants import (NER_NUM_LABELS,
                       NER_DATASET_REPO_ID,
                       NER_MODEL_OUTPUT_DIR,
                       NER_MODEL_CHECKPOINT,
                       NER_ID2LABEL,
                       NER_LABEL2ID)
import evaluate
import numpy as np
from torch import cuda
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NERTrainer")
device = 'cuda' if cuda.is_available() else 'mps'
logger.info(f"Using device: {device}")

class NERTrainer:
    peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS,
                             r=16, # intrinsic rank of trainable weight matrix
                             lora_alpha=32, # similar to learning_rate
                             lora_dropout=0.01, # probability of dropout nodes
                             target_modules=['q_lin', 'k_lin']) # LoRA is applied to query & key layer
    lr = 2e-5
    batch_size = 16
    num_epochs = 6
    weight_decay = 0.01

    # training args
    training_args = TrainingArguments(
        output_dir=NER_MODEL_OUTPUT_DIR,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='./logs',
    )

    def __init__(self):
        self.model = AutoModelForTokenClassification.from_pretrained(
            NER_MODEL_CHECKPOINT,
            num_labels=NER_NUM_LABELS,
            id2label=NER_ID2LABEL,
            label2id=NER_LABEL2ID
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_CHECKPOINT)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = get_peft_model(self.model, self.peft_config).to(device)
        self.model.print_trainable_parameters()


    def _load_data(self):
        return load_dataset(NER_DATASET_REPO_ID)
    
    def _tokenize_function(self, example):
        # tokenize and truncate text
        # self.tokenizer.truncation_side = "right"
        tokenized_inputs = self.tokenizer(
            example['tokens'],
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=64,
        )
        word_ids = tokenized_inputs.word_ids()
        aligned_labels = []

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Special tokens ([CLS], [SEP], etc.)
            elif word_idx != previous_word_idx:
                aligned_labels.append(example['ner_tags'][word_idx])  # Assign the label to the first token of each word
            else:
                aligned_labels.append(-100)  # Subword tokens get label -100

            previous_word_idx = word_idx

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    def preprocess_data(self, dataset_dict):
        return dataset_dict.map(self._tokenize_function)
    
    def postprocess_predictions_and_labels(self, predictions, references):
        true_predictions = []
        true_labels = []
        cmp_count = 0

        for prediction, reference in zip(predictions, references):
            # Only keep labels that are not -100
            true_labels_example = [label for label in reference if label != -100]
            
            # Align predictions: Remove predictions for which the corresponding reference label is -100
            true_predictions_example = [pred for pred, ref in zip(prediction, reference) if ref != -100]

            # Ensure the length of predictions and labels matches
            if len(true_predictions_example) == len(true_labels_example):
                true_labels.append(true_labels_example)
                true_predictions.append(true_predictions_example)
                cmp_count += 1
            else:
                # Log or handle the error (example-level mismatch)
                # print(f"Skipping example due to mismatch: predictions ({len(true_predictions_example)}), labels ({len(true_labels_example)})")
                continue  # Skip this example

        # Flatten the lists (convert from list of lists to a single list)
        true_predictions = [pred for sublist in true_predictions for pred in sublist]
        true_labels = [label for sublist in true_labels for label in sublist]
        logger.info(f"cmp_count = {cmp_count} out of {len(predictions)}")

        return true_predictions, true_labels
    
    def compute_metrics(self, p):
        logits, labels = p
        predictions = np.argmax(logits, axis=1)
        
        # Post-process the predictions and labels to remove -100 values
        true_predictions, true_labels = self.postprocess_predictions_and_labels(predictions, labels)

        # Combine metrics
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")

        # Calculate metrics
        accuracy = accuracy_metric.compute(predictions=true_predictions, references=true_labels)
        precision = precision_metric.compute(predictions=true_predictions, references=true_labels, average="weighted")
        recall = recall_metric.compute(predictions=true_predictions, references=true_labels, average="weighted")
        f1 = f1_metric.compute(predictions=true_predictions, references=true_labels, average="weighted")

        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"]
        }
    
    def run(self, tokenized_dataset):
        trainer = Trainer(
            model=self.model,
            args=self.training_args, # Hyperparamaters
            train_dataset=tokenized_dataset["train"], # training data
            eval_dataset=tokenized_dataset["validation"], # validation data
            tokenizer=self.tokenizer, # tokenizer
            compute_metrics=self.compute_metrics,  # model perfomance evaluation metric
        )
        trainer.train()


if __name__ == '__main__':
    nerTrainer = NERTrainer()
    dataset_dict = nerTrainer._load_data()
    tokenized_dataset = nerTrainer.preprocess_data(dataset_dict)
    logger.info(tokenized_dataset)
    nerTrainer.run(tokenized_dataset)

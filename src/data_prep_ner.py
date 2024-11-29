# This module handles the data preparation for intent classification task
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
from constants import NER_LABELLED_DATA_PATH, NER_DATASET_REPO_ID
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataPrepNER")

class DataPrepNER:
    
    def _read_data(self):
        return pd.read_parquet(NER_LABELLED_DATA_PATH)


    def get_data(self):
        """Get the labelled data"""
        return self._read_data()
    
    def prepare_train_test_datasets(self, df):
        """Split the train test split dataframs and convert them to datasets"""
        train_df, val_df = train_test_split(df, test_size=0.0111, random_state=42)
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def upload_to_hf(self, dataset_dict, repo_id):
        """Upload the dataset to Hugging Face Hub."""
        token = os.getenv("HF_TOKEN")
        if token is not None:
            dataset_dict.push_to_hub(repo_id, token=token)
            logger.info(f"Uploaded DatasetDict to Hugging Face Hub at {repo_id}")
        else:
            logger.error("Cannot upload to HF as HF_TOKEN was not in env")

    def run_pipeline(self):
        """Run the complete data preparation pipeline."""
        logger.info("Starting the data preparation pipeline...")
        
        # Load and sample the data
        data = self.get_data()
        
        # Split into train and validation sets
        dataset_dict = self.prepare_train_test_datasets(data)
        logger.info(dataset_dict)
        
        # Upload the dataset to Hugging Face Hub
        self.upload_to_hf(dataset_dict, NER_DATASET_REPO_ID)
        
        logger.info("Pipeline completed successfully.")


if __name__ == '__main__':
    ner_data_prep = DataPrepNER()
    ner_data_prep.run_pipeline()

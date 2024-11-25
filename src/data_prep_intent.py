# This module handles the data preparation for intent classification task
import os
import pandas as pd
from constants import INTENT_LABEL2ID, INTENT_DATASET_REPO_ID, INTENT_LABELLED_DATA_PATH
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataPrepIntent")

class DataPrepIntent:
    
    DFLT_PCT_VAL = 1.0
    SAMPLING_PERCENTAGES = {
        'information_intent': 1.0,   # 100% sampling for information_intent
        'yelp_intent': 1.0,          # 100% sampling for yelp_intent
        'weather_intent': 1.0,       # 100% sampling for weather_intent
        'navigation_intent': 1.0,    # 100% sampling for navigation_intent
        'purchase_intent': 1.0,      # 100% sampling for purchase_intent
        'translation_intent': 1.0,   # 100% sampling for translation_intent
        'travel_intent': 1.0,        # 100% sampling for travel_intent
        'unknown': 1.0               # 100% sampling for unknown
    }

    def _read_data(self):
        return pd.read_csv(INTENT_LABELLED_DATA_PATH)

    def _sample_tgt_by_percentages(self, df):
        """return sampled label data by sampling percentages"""
        sampled_df = df.groupby('target', group_keys=False).apply(
            lambda x: x.sample(frac=self.SAMPLING_PERCENTAGES.get(x.name, self.DFLT_PCT_VAL),
                               random_state=42), include_groups=True)\
                       .reset_index(drop=True)
        sampled_df['label'] = sampled_df['target'].map(INTENT_LABEL2ID)
        logger.info("Size of the sampled data = %d", len(sampled_df))
        logger.info("Sampled target sizes %s", sampled_df['label'].value_counts())
        return sampled_df

    def get_data(self):
        """Get the labelled data"""
        return self._sample_tgt_by_percentages(self._read_data())
    
    def prepare_train_test_datasets(self, df):
        """Split the train test split dataframs and convert them to datasets"""
        train_df, val_df = train_test_split(df, test_size=0.05, random_state=42, stratify=df['label'])
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
        
        # Upload the dataset to Hugging Face Hub
        self.upload_to_hf(dataset_dict, INTENT_DATASET_REPO_ID)
        
        logger.info("Pipeline completed successfully.")


if __name__ == '__main__':
    intent_data_prep = DataPrepIntent()
    intent_data_prep.run_pipeline()

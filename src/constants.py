## used by intent modules
INTENT_LABELLED_DATA_VERSION = "v4"
INTENT_LABELLED_DATA_PATH = f"https://raw.githubusercontent.com/mozilla/smart_intent/refs/heads/main/data/marco_train_{INTENT_LABELLED_DATA_VERSION}.csv"
INTENT_ID2LABEL = {0: 'information_intent',
                   1: 'yelp_intent',
                   2: 'navigation_intent',
                   3: 'travel_intent',
                   4: 'purchase_intent',
                   5: 'weather_intent',
                   6: 'translation_intent',
                   7: 'unknown'}
INTENT_LABEL2ID = {label:id for id,label in INTENT_ID2LABEL.items()}
NUM_LABELS = len(INTENT_ID2LABEL)
INTENT_DATASET_REPO_ID = "Mozilla/smart_intent_dataset"
INTENT_MODEL_CHECKPOINT = 'google/mobilebert-uncased'
INTENT_MODEL_OUTPUT_DIR = "models/mobilebert-uncased" + "-lora-intent-classification-v3"
MODEL_ARTIFACT_DIR = "tmp/mobilebert_lora_combined_model/"
## Make sure you change to correct Repo id ("Mozilla/mobilebert-uncased-finetuned-LoRA-intent-classifier")
INTENT_MODEL_REPO_ID = "chidamnat2002/smart_intent_model" 

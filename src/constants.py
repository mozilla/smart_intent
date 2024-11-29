## used by intent modules
INTENT_LABELLED_DATA_VERSION = "v6"
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
INTENT_NUM_LABELS = len(INTENT_ID2LABEL)
INTENT_DATASET_REPO_ID = "Mozilla/smart_intent_dataset"
INTENT_MODEL_CHECKPOINT = 'google/mobilebert-uncased'
INTENT_MODEL_OUTPUT_DIR = "models/mobilebert-uncased" + f"-lora-intent-classification-{INTENT_LABELLED_DATA_VERSION}"
MODEL_ARTIFACT_DIR = "tmp/mobilebert_lora_combined_model/"
## Make sure you change to correct Repo id ("Mozilla/mobilebert-uncased-finetuned-LoRA-intent-classifier")
INTENT_MODEL_REPO_ID = "chidamnat2002/smart_intent_model"

## used by NER modules
NER_LABELLED_DATA_VERSION = "v6"
NER_LABELLED_DATA_PATH = f"https://raw.githubusercontent.com/mozilla/smart_intent/refs/heads/main/data/synthetic_loc_dataset_{NER_LABELLED_DATA_VERSION}.parquet"
NER_ID2LABEL = {
    0: "O",        # Outside any named entity
    1: "B-PER",    # Beginning of a person entity
    2: "I-PER",    # Inside a person entity
    3: "B-ORG",    # Beginning of an organization entity
    4: "I-ORG",    # Inside an organization entity
    5: "B-CITY",    # Beginning of a city entity
    6: "I-CITY",    # Inside a city entity
    7: "B-STATE",    # Beginning of a state entity
    8: "I-STATE",    # Inside a state entity
    9: "B-CITYSTATE",   # Beginning of a city_state entity
   10: "I-CITYSTATE",   # Inside a city_state entity
}

NER_LABEL2ID = {v: k for k, v in NER_ID2LABEL.items()}
NER_NUM_LABELS = 11
NER_DATASET_REPO_ID = "Mozilla/smart_ner_dataset"
NER_MODEL_CHECKPOINT = 'distilbert/distilbert-base-uncased'
NER_MODEL_OUTPUT_DIR = f"models/distilbert-uncased-NER-LoRA-{NER_LABELLED_DATA_VERSION}"
NER_MODEL_ARTIFACT_DIR = "tmp/merged_distilbert_uncased_ner/"
## Make sure you change to correct Repo id ("Mozilla/distilbert-uncased-NER-LoRA")
NER_MODEL_REPO_ID = "chidamnat2002/smart_ner_model"
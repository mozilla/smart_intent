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
INTENT_REPO_ID = "Mozilla/smart_intent_dataset"

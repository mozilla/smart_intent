from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class IntentClassifier:
    def __init__(self):
        self.id2label = {0: 'information_intent',
                    1: 'yelp_intent',
                    2: 'navigation_intent',
                    3: 'travel_intent',
                    4: 'purchase_intent',
                    5: 'weather_intent',
                    6: 'translation_intent',
                    7: 'unknown'}
        self.label2id = {label:id for id,label in self.id2label.items()}

        self.tokenizer = AutoTokenizer.from_pretrained("chidamnat2002/intent_classifier")
        self.intent_model = AutoModelForSequenceClassification.from_pretrained('chidamnat2002/intent_classifier',
                                                                        num_labels=8,
                                                                        torch_dtype=torch.bfloat16,
                                                                        id2label=self.id2label,
                                                                        label2id=self.label2id)

    def find_intent(self, sequence, verbose=False):
        inputs = self.tokenizer(sequence,
                return_tensors="pt",  # ONNX requires inputs in NumPy format
                padding="max_length",  # Pad to max length
                truncation=True,       # Truncate if the text is too long
                max_length=64)

        self.intent_model.eval()
        with torch.no_grad():
            outputs = self.intent_model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1)
            rounded_probabilities = torch.round(probabilities, decimals=3)
            
            pred_result = self.id2label[prediction]
            proba_result = dict(zip(self.label2id.keys(), rounded_probabilities.tolist()[0]))
            if verbose:
                print(sequence + " -> " + pred_result)
                print(proba_result,  "\n")
            return pred_result, proba_result


def main():
    text_list = [
        'floor repair cost', 
        'pet store near me', 
        'who is the us president', 
        'italian food',
        'sandwiches for lunch',
        "cheese burger cost",
        "What is the weather today",
        "what is the capital of usa",
        "cruise trip to carribean",
    ]
    cls = IntentClassifier()
    for sequence in text_list:
        cls.find_intent(sequence)

if __name__ == '__main__':
    main()
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import requests
import os

class LocationFinder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Mozilla/distilbert-uncased-NER-LoRA")
        model_url = "https://huggingface.co/Mozilla/distilbert-uncased-NER-LoRA/resolve/main/onnx/model_quantized.onnx"
        model_dir_path = "models"
        model_path = f"{model_dir_path}/distilbert-uncased-NER-LoRA"
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        if not os.path.exists(model_path):
            print("Downloading ONNX model...")
            response = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(response.content)
            print("ONNX model downloaded.")

        # Load the ONNX model
        self.ort_session = ort.InferenceSession(model_path)

    def find_location(self, sequence, verbose=False):
        inputs = self.tokenizer(sequence,
                                return_tensors="np",  # ONNX requires inputs in NumPy format
                                padding="max_length",  # Pad to max length
                                truncation=True,       # Truncate if the text is too long
                                max_length=64)
        input_feed = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64),
        }
        
        # Run inference with the ONNX model
        outputs = self.ort_session.run(None, input_feed)
        logits = outputs[0]  # Assuming the model output is logits
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        
        predicted_ids = np.argmax(logits, axis=-1)
        predicted_probs = np.max(probabilities, axis=-1)
        
        # Define the threshold for NER probability
        threshold = 0.6
        
        label_map = {
            0: "O",        # Outside any named entity
            1: "B-PER",    # Beginning of a person entity
            2: "I-PER",    # Inside a person entity
            3: "B-ORG",    # Beginning of an organization entity
            4: "I-ORG",    # Inside an organization entity
            5: "B-LOC",    # Beginning of a location entity
            6: "I-LOC",    # Inside a location entity
            7: "B-MISC",   # Beginning of a miscellaneous entity
            8: "I-MISC"    # Inside a miscellaneous entity
        }
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # List to hold the detected location terms
        location_entities = []
        current_location = []

        # Loop through each token and its predicted label and probability
        for i, (token, predicted_id, prob) in enumerate(zip(tokens, predicted_ids[0], predicted_probs[0])):
            label = label_map[predicted_id]

            # Ignore special tokens like [CLS], [SEP]
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
        
            # Only consider tokens with probability above the threshold
            if prob > threshold:
                # If the token is a part of a location entity (B-LOC or I-LOC)
                if label in ["B-LOC", "I-LOC"]:
                    if label == "B-LOC":
                        # If we encounter a B-LOC, we may need to store the previous location
                        if current_location:
                            location_entities.append(" ".join(current_location).replace("##", ""))
                        # Start a new location entity
                        current_location = [token]
                    elif label == "I-LOC" and current_location:
                        # Continue appending to the current location entity
                        current_location.append(token)
                else:
                    # If we encounter a non-location entity, store the current location and reset
                    if current_location:
                        location_entities.append(" ".join(current_location).replace("##", ""))
                        current_location = []
        
        # Append the last location entity if it exists
        if current_location:
            location_entities.append(" ".join(current_location).replace("##", ""))

        # Return the detected location terms
        return location_entities[0] if location_entities != [] else None


if __name__ == '__main__':
    query = "weather in seattle"
    loc_finder = LocationFinder()
    location = loc_finder.find_location(query)
    print(f"query = {query} => {location}")


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
        
        # Define the label map for city, state, citystate, etc.
        label_map = {
            0: "O",        # Outside any named entity
            1: "B-PER",    # Beginning of a person entity
            2: "I-PER",    # Inside a person entity
            3: "B-ORG",    # Beginning of an organization entity
            4: "I-ORG",    # Inside an organization entity
            5: "B-CITY",   # Beginning of a city entity
            6: "I-CITY",   # Inside a city entity
            7: "B-STATE",  # Beginning of a state entity
            8: "I-STATE",  # Inside a state entity
            9: "B-CITYSTATE",   # Beginning of a city_state entity
        10: "I-CITYSTATE",   # Inside a city_state entity
        }
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Initialize lists to hold detected entities
        city_entities = []
        state_entities = []
        city_state_entities = []
        
        for token, predicted_id, prob in zip(tokens, predicted_ids[0], predicted_probs[0]):
            if prob > threshold:
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                if label_map[predicted_id] in ["B-CITY", "I-CITY"]:
                    # Handle the case of continuation tokens (like "##" in subwords)
                    if token.startswith("##") and city_entities:
                        city_entities[-1] += token[2:]  # Remove "##" and append to the last token
                    else:
                        city_entities.append(token)
                elif label_map[predicted_id] in ["B-STATE", "I-STATE"]:
                    if token.startswith("##") and state_entities:
                        state_entities[-1] += token[2:]
                    else:
                        state_entities.append(token)
                elif label_map[predicted_id] in ["B-CITYSTATE", "I-CITYSTATE"]:
                    if token.startswith("##") and city_state_entities:
                        city_state_entities[-1] += token[2:]
                    else:
                        city_state_entities.append(token)

        # Combine city_state entities and split into city and state if necessary
        if city_state_entities:
            city_state_str = " ".join(city_state_entities)
            city_state_split = city_state_str.split(",")  # Split on comma to separate city and state
            city_res = city_state_split[0].strip() if city_state_split[0] else None
            state_res = city_state_split[1].strip() if len(city_state_split) > 1 else None
        else:
            # If no city_state entities, use detected city and state entities separately
            city_res = " ".join(city_entities).strip() if city_entities else None
            state_res = " ".join(state_entities).strip() if state_entities else None

        # Return the detected city and state as separate components
        return {
            'city': city_res,
            'state': state_res
        }


if __name__ == '__main__':
    query = "weather in san francisco, ca"
    loc_finder = LocationFinder()
    entities = loc_finder.find_location(query)
    print(f"query = {query} => {entities}")

import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import requests
import os
import torch.nn.functional as F
import torch
import shap
import pandas as pd

class LocationFinder:
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

    def preprocess(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        return inputs["input_ids"].numpy(), inputs["attention_mask"].numpy()
        
    def predict(self, inputs):
        input_ids, attention_mask = inputs
        # Run inference
        outputs = self.ort_session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        return outputs[0]

    def predict_proba(self, texts):
        # Set max_length to the model's maximum output sequence length
        max_length = 64
        num_classes = self.ort_session.get_outputs()[0].shape[-1]  # Number of classes from the model

    
        # Initialize 3D array for all probabilities with zeros (padding)
        all_probs = np.zeros((len(texts), max_length, num_classes))
        
        for i, text in enumerate(texts):
            # Tokenize and prepare input
            input_ids, attention_mask = self.preprocess(text)
            
            # Run model inference to get logits
            logits = self.predict((input_ids, attention_mask))
            
            # Apply softmax to get probabilities for each token
            prob = F.softmax(torch.tensor(logits), dim=-1).numpy()
            
            # Trim to actual token length, ignoring padding tokens
            tokens = self.tokenizer(text, truncation=True, padding="max_length")['input_ids']
            token_probabilities = prob[:len(tokens), :]
            
            # Copy token probabilities into the pre-initialized array up to token length
            all_probs[i, :len(tokens), :] = token_probabilities  # Fill up to token length
        
        # Return the padded 3D numpy array
        return all_probs
    
    def shap_predict_wrapper(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.flatten().tolist()  # Ensure it’s a list of strings
        probs = self.predict_proba(texts)
        # Flatten output to ensure it's a 2D array
        return np.array([p.flatten() for p in probs])
    
    def show_explanation(self, query):
        # Convert the background text to a 2D numpy array
        background_text = np.array([["This is a sample background text for SHAP."]])

        # Initialize KernelExplainer with the 2D numpy array background
        explainer = shap.KernelExplainer(self.shap_predict_wrapper, background_text)
        text_to_explain = np.array([query])
        # Generate SHAP values for the tokens
        shap_values = explainer.shap_values(text_to_explain)
        num_tokens = 64  # Adjust if your input length is different
        num_classes = 11
        reshaped_shap_values = shap_values[0].reshape(num_tokens, num_classes)
        # aggregated_shap_values = reshaped_shap_values.sum(axis=1)
        masks = [idx for idx, mask in enumerate(self.tokenizer(text_to_explain[0], truncation=True, padding="max_length", max_length=64)['attention_mask']) if mask]
        text_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer(text_to_explain[0], truncation=True, padding="max_length", max_length=64)['input_ids'])
        trimmed_shap_values = reshaped_shap_values[:len(masks), :]
        shap_values_by_class = pd.DataFrame(trimmed_shap_values, index=text_tokens[:len(masks)], columns=list(self.label_map.values()))
        return shap_values_by_class


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
        threshold = 0.5
        
        label_map = self.label_map
        
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

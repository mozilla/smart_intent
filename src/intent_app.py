import streamlit as st
import streamlit.components.v1 as components
from infer_intent import IntentClassifier
from infer_location import LocationFinder
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Intent classifier")

@st.cache_resource
def get_intent_classifier():
    cls = IntentClassifier()
    return cls

@st.cache_resource
def get_location_finder():
    ner = LocationFinder()
    return ner

cls = get_intent_classifier()
query = st.text_input("Enter a query", value="What is the weather today")
query = query.lower()
pred_result, proba_result = cls.find_intent(query)

ner = get_location_finder()
location = ner.find_location(query)

st.markdown(f"Intent = :green[{pred_result}]")
st.markdown(f"Location = :green[{location}]")
keys = list(proba_result.keys())
values = list(proba_result.values())

# Creating the bar plot
fig, ax = plt.subplots()
ax.barh(keys, values)

# Adding labels and title
ax.set_xlabel('probability score')
ax.set_ylabel('Intents')
ax.set_title('Intents probability score')

col1, col2 = st.columns([2,4])

with col1:
    st.pyplot(fig)

with col2:
    exp3 = st.expander("NER SHAP values by tokens and class")
    with exp3:
        shap_values_by_class = ner.show_explanation(query)
        plt.figure(figsize=(10, 8))
        sns.heatmap(shap_values_by_class, cmap='RdBu_r', center=0, annot=True, fmt=".2f", cbar_kws={'label': 'SHAP Value'})
        plt.xlabel("Entity Class")
        plt.ylabel("Tokens")
        plt.title("SHAP Values by Token and Entity Class")
        st.pyplot(plt)

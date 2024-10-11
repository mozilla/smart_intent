import streamlit as st
import streamlit.components.v1 as components
from infer_intent import IntentClassifier
from infer_location import LocationFinder
import matplotlib.pyplot as plt

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
    exp = st.expander("Explore training data")
    with exp:
        html_file = "reports/web_search_intents.html"
        with open(html_file, 'r', encoding='utf-8') as f:
            plotly_html = f.read()
            components.html(plotly_html, height=900, width=900)

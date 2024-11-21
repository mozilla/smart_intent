import streamlit as st
# import streamlit.components.v1 as components
from infer_intent import IntentClassifier
from infer_location import LocationFinder
import matplotlib.pyplot as plt
import seaborn as sns
import json

st.set_page_config(layout="wide")
st.title("Intent classifier")

PREPOSITIONS = ["in", "at", "on", "for", "to", "near"]

@st.cache_resource
def get_intent_classifier():
    cls = IntentClassifier()
    return cls

@st.cache_resource
def get_location_finder():
    ner = LocationFinder()
    return ner

@st.cache_data
def get_geonames_city_state_data():
    geonames_file = "data/geonames-cities-states.json"
    with open(geonames_file, 'r') as f:
        geonames_dict = json.load(f)
    states_lkp = set([state_cd.lower() for state_cd in list(geonames_dict['states_by_abbr'].keys())] + [state['name'].lower() for state in geonames_dict['states_by_abbr'].values()])
    city_lkp = set([city_name  for city in geonames_dict['cities'] for city_name in city['alternate_names']])
    return city_lkp, states_lkp


def check_location_in_query(query, city_lkp, state_lkp):
    words = [word for word in query.replace(',',' ').split() if word not in PREPOSITIONS]
    return any([word for word in words if len(word) > 1 and (word in state_lkp or word in city_lkp)])

cls = get_intent_classifier()
query = st.text_input("Enter a query", value="What is the weather today")
query = query.lower()
pred_result, proba_result = cls.find_intent(query)

city_lkp, states_lkp = get_geonames_city_state_data()
ner = get_location_finder()
if check_location_in_query(query, city_lkp, states_lkp):
    location = ner.find_location(query)
else:
    location = {'city': None, 'state': None}

st.markdown(f"Intent = :green[{pred_result}]")
st.markdown(f"Location = :green[{location}]")
st.write(proba_result)
st.markdown(f"probability = :green[{proba_result[pred_result]}]")
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

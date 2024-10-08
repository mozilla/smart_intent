## smart_intent
Classify query to various intents as below
1) `information_intent`: intention to learn more about something. Eg. what is programming?, How does docking station work ?
2) `yelp_intent`: intention to look for services/local businesses. Eg. italian food near me, floor repair cost
3) `navigation_intent`: intention to navigate from search to the website. Eg. email login, bank branch routing number
4) `travel_intent`: intention to travel. Eg. hotels in Paris, cruise trip to carribean islands
5) `purchase_intent`: intention to purchase products. Eg. price of us open tennis tickets, buy iphone 10
6) `weather_intent`: to know the weather or temperature of a city. Eg. weather in Miami, temperature in Seattle
7) `translation_intent`: to translate from and to various languages. Eg. what is hello in spanish, translate hi to japanese
8) `unknown`: when the intent is inconclusive

## Steps to install the App 
1) clone the repo
2) python -m venv venv
3) source /venv/bin/activate
4) python -m pip install -r requirements
5) streamlit run src/intent_app.py

<img width="1586" alt="image" src="https://github.com/user-attachments/assets/c5b1931c-fa9a-47ad-a229-412610b83910">


## steps to run the sample intent inference
steps 1 - 4 from above
5) python src/infer_intent.py

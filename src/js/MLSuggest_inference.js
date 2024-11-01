/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * MLSuggest_inference.js helps infer MLSuggest intents and locations 
 * (city and state).
 */

ChromeUtils.defineESModuleGetters(this, {
    MLSuggest: "resource:///modules/urlbar/private/MLSuggest.sys.mjs",
  });
  
  const YELP_KEYWORDS_DATA = "https://firefox-settings-attachments.cdn.mozilla.net/main-workspace/quicksuggest/33987d71-9e87-4b7e-86d3-6f292b89e8bf.json";
  const YELP_VAL_DATA = "https://raw.githubusercontent.com/mozilla/smart_intent/refs/heads/main/data/yelp_val_generated_data.json";
  
  const NER_VAL_DATA = "https://raw.githubusercontent.com/mozilla/smart_intent/refs/heads/main/data/named_entity_val_generated_data.json";
  
  const type = "NER_VAL_DATA"
  
  // Get the user's default download directory
  const OUTPUT_FILE_PATH = `${Services.dirsvc.get("DfltDwnld", Ci.nsIFile).path}/ML_output_${type}.json`;
  
  async function get_yelp_keywords() {
    const response = await fetch(YELP_KEYWORDS_DATA);
    if (!response.ok) {
      throw new Error(`Failed to fetch data: ${response.statusText}`);
    }
  
    // Parse the JSON response
    const data = await response.json();
    return data[0].subjects;
  }
  
  async function get_yelp_val_data() {
    const response = await fetch(YELP_VAL_DATA);
    if (!response.ok) {
      throw new Error(`Failed to fetch data: ${response.statusText}`);
    }
  
    // Parse the JSON response
    const data = await response.json();
    return data.queries;
  }
  
  async function get_ner_val_data() {
    const response = await fetch(NER_VAL_DATA);
    if (!response.ok) {
      throw new Error(`Failed to fetch data: ${response.statusText}`);
    }
  
    // Parse the JSON response
    const data = await response.json();
    return data.queries;
  }
  
  async function fetchAndProcessData() {
    try {
      // Fetch the JSON data
      let queries;
      if (type === "YELP_KEYWORDS_DATA") {
        queries = await get_yelp_keywords();
      } else if (type === "YELP_VAL_DATA") {
        queries = await get_yelp_val_data();
      } else if (type === "NER_VAL_DATA") {
        queries = await get_ner_val_data();
      }
  
  
      // Ensure MLSuggest is initialized
      await MLSuggest.initialize();
  
      // Process each subject and collect results
      const results = [];
      for (const query of queries) {
        const suggestion = await MLSuggest.makeSuggestions(query);
        const res = {
          query,
          intent: suggestion.intent,
          city: suggestion.location.city,
          state: suggestion.location.state
        };
        results.push(res);
      }
  
      // Write results to a file
      await writeResultsToFile(results);
      console.log("Processing completed. Results saved to:", OUTPUT_FILE_PATH);
  
    } catch (error) {
      console.error("Error processing data:", error);
    } finally {
      await MLSuggest.shutdown();
    }
  }
  
  // Utility to write results to a local JSON file using IOUtils
  async function writeResultsToFile(results) {
    try {
      const json = JSON.stringify(results, null, 2);
      await IOUtils.writeUTF8(OUTPUT_FILE_PATH, json);
      console.log("Results successfully written to:", OUTPUT_FILE_PATH);
    } catch (error) {
      console.error("Failed to write results to file:", error);
    }
  }
  
  // Run the fetch and process function
  fetchAndProcessData();
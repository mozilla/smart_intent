import MLSuggest from './MLSuggest.sys.mjs';

async function testMLSuggest(query) {
    // Initialize the ML models (optional if models need to be loaded in advance)
    await MLSuggest.initialize();

    // Make ML suggestions for a query
    const result = await MLSuggest.makeMLSuggestions(query);
    console.log(result);

    // // Shutdown the engines after use
    // await MLSuggest.shutdown();
}

query = "restaurants in seattle"
testMLSuggest(query);

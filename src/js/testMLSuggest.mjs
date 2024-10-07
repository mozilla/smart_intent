import MLSuggest from './MLSuggest.sys.mjs';

async function testMLSuggest() {
    const mlSuggest = new MLSuggest();

    // // Initialize the ML models (optional if models need to be loaded in advance)
    await mlSuggest.initialize();

    // Make ML suggestions for a query
    const result = await mlSuggest.makeMLSuggestions("restaurants in seattle");
    console.log(result);

    // Shutdown the engines after use
    await mlSuggest.shutdown();
}

testMLSuggest();

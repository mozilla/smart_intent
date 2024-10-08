import mlSuggestInstance from './MLSuggest.sys.mjs';

async function testMLSuggest() {
    // Initialize the ML models (optional if models need to be loaded in advance)
    await mlSuggestInstance.initialize();

    // Make ML suggestions for a query
    const result = await mlSuggestInstance.makeMLSuggestions("restaurants in seattle");
    console.log(result);

    // // Shutdown the engines after use
    // await mlSuggestInstance.shutdown();
}

testMLSuggest();

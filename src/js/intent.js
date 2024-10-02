// cache to store initalized engine
var modelEngines = {};


// Initialize the ML model based on given options
async function initializeMLModel(options) {
    const engine_id = `${options.taskName}-${options.modelId}`;

    // if engine was cached previously
    if (modelEngines[engine_id]) {
        console.log(`Resusing cached engine for model: ${options.modelId}`);
        return modelEngines[engine_id];
    }

    // if engine was not cached, initialize the engine
    const { createEngine } = ChromeUtils.importESModule("chrome://global/content/ml/EngineProcess.sys.mjs");

    try {
        const engine = await createEngine(options);
        console.log("Engine successfully created:", engine);
        return engine;
    } catch (error) {
        console.error("Error creating engine:", error);
        throw error;
    }
}

// Find the intent of the query using the intent classifier model
async function findIntent(query) {
    const optionsForIntent = {
        taskName: "text-classification",
        modelId: "chidamnat2002/intent_classifier",
        modelRevision: "main",
        quantization: "q8"
    };

    const engineIntentClassifier = await initializeMLModel(optionsForIntent);
    const request = { args: [query], options: {} };
    const res = await engineIntentClassifier.run(request);

    return res[0].label;  // Return the first label from the result
}

// Find named entities in the query using the NER model
async function findNER(query) {
    const optionsForNER = {
        taskName: "token-classification",
        modelId: "Xenova/bert-base-NER",
        modelRevision: "main",
        quantization: "q8"
    };

    const engineNER = await initializeMLModel(optionsForNER);
    const request = { args: [query], options: {} };
    const res = await engineNER.run(request);

    return res;
}

// Combine location entities (B-LOC and I-LOC) from NER results
async function combineLocations(nerResult, nerThreshold) {
    let locResult = [];

    nerResult.forEach(res => {
        if ((res.entity === 'B-LOC' || res.entity === 'I-LOC') && res.score > nerThreshold) {
            if (res.word.startsWith('##') && locResult.length > 0) {
                locResult[locResult.length - 1] += res.word.slice(2);  // Append to last word
            } else {
                locResult.push(res.word);
            }
        }
    });

    return locResult.length > 0 ? locResult.join(' ') : null;
}

// Convert text to Title Case
function toTitleCase(text) {
    return text.toLowerCase().split(' ').map(word =>
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

function findSubjectFromQuery(query, location) {
    const queryLowerCase = query.toLowerCase();
    const locationLowerCase = location.toLowerCase();

    const subjectWithoutLocation = queryLowerCase.replace(locationLowerCase, '').trim();

    // Clean the subject by removing any dangling prepositions
    return cleanSubject(toTitleCase(subjectWithoutLocation));
}

function cleanSubject(subject) {
    const prepositions = ['in', 'at', 'on', 'for', 'to'];
    const words = subject.split(' ');
    if (prepositions.includes(words[words.length - 1].toLowerCase())) {
        words.pop();  // Remove trailing preposition
    }
    return words.join(' ').trim();
}

function sumObjectsByKey(...objs) {
    return objs.reduce((a, b) => {
        for (let k in b) {
            if (b.hasOwnProperty(k))
                a[k] = (a[k] || 0) + b[k];
        }
      return a;
    }, {});
}

// Main execution
(async () => {
    const NER_THRESHOLD = 0.5;
    // const query = "san antonio weather";
    const query = "restaurants in seattle";
    const intentRes = await findIntent(query);
    const nerResult = await findNER(toTitleCase(query));
    const locationResVal = await combineLocations(nerResult, NER_THRESHOLD);
    const subjectRes = findSubjectFromQuery(query, locationResVal)
  
    const finalRes = {
        intent: intentRes,
        location: locationResVal,
        subject: subjectRes
    };
        
    finalRes.metrics = sumObjectsByKey(nerResult.metrics, nerResult.metrics);

    console.log("Final result:", finalRes);
})();

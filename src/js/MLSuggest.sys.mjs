/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 *
 */
class MLSuggest {
  #modelEngines = {};

  // Private constructor to prevent instantiation from outside
  constructor() {
    if (MLSuggest.instance) {
      return MLSuggest.instance;
    }
    MLSuggest.instance = this;
  }

  // Internal function to initialize the ML model (private)
  async #initializeMLModel(options) {
    const engine_id = `${options.taskName}-${options.modelId}`;

    // If engine was cached previously
    if (this.#modelEngines[engine_id]) {
      return this.#modelEngines[engine_id];
    }

    // If engine was not cached, initialize the engine
    const { createEngine } = ChromeUtils.importESModule(
      "chrome://global/content/ml/EngineProcess.sys.mjs"
    );

    options.engineId = engine_id;
    options.numThreads = 4;
    try {
      const engine = await createEngine(options);
      this.#modelEngines[engine_id] = engine; // Cache the engine
      return engine;
    } catch (error) {
      console.error("Error creating engine:", error);
      throw error;
    }
  }

  // Internal function to find intent
  async #findIntent(query) {
    const optionsForIntent = {
      taskName: "text-classification",
      modelId: "mozilla/mobilebert-uncased-finetuned-LoRA-intent-classifier",
      modelRevision: "main",
      quantization: "q8",
    };

    const engineIntentClassifier = await this.#initializeMLModel(
      optionsForIntent
    );
    const request = { args: [query], options: {} };
    const res = await engineIntentClassifier.run(request);

    return res[0].label; // Return the first label from the result
  }

  // Internal function to find named entities
  async #findNER(query) {
    const optionsForNER = {
      taskName: "token-classification",
      modelId: "mozilla/distilbert-NER-LoRA",
      modelRevision: "main",
      quantization: "q8",
    };

    const engineNER = await this.#initializeMLModel(optionsForNER);
    const request = { args: [query], options: {} };
    const res = await engineNER.run(request);

    return res;
  }

  // Internal helper function to combine locations
  async #combineLocations(nerResult, nerThreshold) {
    let locResult = [];

    nerResult.forEach(res => {
      if (
        (res.entity === "B-LOC" || res.entity === "I-LOC") &&
        res.score > nerThreshold
      ) {
        if (res.word.startsWith("##") && locResult.length) {
          locResult[locResult.length - 1] += res.word.slice(2); // Append to last word
        } else {
          locResult.push(res.word);
        }
      }
    });

    return locResult.length ? locResult.join(" ") : null;
  }

  #findSubjectFromQuery(query, location) {
    // If location is null, return the entire query as the subject
    if (!location) {
      return query;
    }

    const queryLowerCase = query.toLowerCase();
    const locationLowerCase = location.toLowerCase();
    const subjectWithoutLocation = queryLowerCase
      .replace(locationLowerCase, "")
      .trim();
    return this.#cleanSubject(subjectWithoutLocation);
  }

  #cleanSubject(subject) {
    const prepositions = ["in", "at", "on", "for", "to"];
    const words = subject.split(" ");
    if (prepositions.includes(words[words.length - 1].toLowerCase())) {
      words.pop(); // Remove trailing preposition
    }
    return words.join(" ").trim();
  }

  #sumObjectsByKey(...objs) {
    return objs.reduce((a, b) => {
      for (let k in b) {
        if (b.hasOwnProperty(k)) a[k] = (a[k] || 0) + b[k];
      }
      return a;
    }, {});
  }

  // Make ML-based suggestions
  async makeMLSuggestions(query) {
    const NER_THRESHOLD = 0.5;

    try {
      // Trim the query to remove leading/trailing whitespace
      const trimmedQuery = query.trim();

      const [intentRes, nerResult] = await Promise.all([
        this.#findIntent(trimmedQuery),
        this.#findNER(trimmedQuery),
      ]);

      const locationResVal = await this.#combineLocations(
        nerResult,
        NER_THRESHOLD
      );
      const subjectRes = this.#findSubjectFromQuery(
        trimmedQuery,
        locationResVal
      );

      const finalRes = {
        intent: intentRes,
        location: locationResVal,
        subject: subjectRes,
      };

      finalRes.metrics = this.#sumObjectsByKey(
        nerResult.metrics,
        nerResult.metrics
      );

      // console.log("Final result:", finalRes);
      return finalRes;
    } catch (error) {
      console.error("Error in processing:", error);
      return null;
    }
  }

  // Shutdown engines
  async shutdown() {
    await Promise.all(
      Object.values(this.#modelEngines).map(async engine => {
        if (engine.terminate) {
          await engine.terminate();
        }
      })
    );
  }

  // Initialize engines (if needed for warm start)
  async initialize() {
    const optionsForIntent = {
      taskName: "text-classification",
      modelId: "mozilla/mobilebert-uncased-finetuned-LoRA-intent-classifier",
      modelRevision: "main",
      quantization: "q8",
    };

    const optionsForNER = {
      taskName: "token-classification",
      modelId: "mozilla/distilbert-NER-LoRA",
      modelRevision: "main",
      quantization: "q8",
    };

    // Preload models here if necessary
    await this.#initializeMLModel(optionsForIntent);
    await this.#initializeMLModel(optionsForNER);
  }
}

// Export the singleton instance
const mlSuggestInstance = new MLSuggest();
export default mlSuggestInstance;

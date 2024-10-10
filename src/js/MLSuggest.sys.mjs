/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * MLSuggest helps with ML based suggestions around intents and location. 
 */

const lazyModules = {};

ChromeUtils.defineESModuleGetters(lazyModules, {
  createEngine: "chrome://global/content/ml/EngineProcess.sys.mjs",
});

/**  
 * These INTENT_OPTIONS and NER_OPTIONS will go to remote setting server and depends
 * on https://bugzilla.mozilla.org/show_bug.cgi?id=1923553 
  */
const INTENT_OPTIONS = {
  taskName: "text-classification",
  modelId: "mozilla/mobilebert-uncased-finetuned-LoRA-intent-classifier",
  modelRevision: "main",
  dtype: "q8",
};

const NER_OPTIONS = {
  taskName: "token-classification",
  modelId: "mozilla/distilbert-NER-LoRA",
  modelRevision: "main",
  dtype: "q8",
};

const NER_THRESHOLD = 0.5;

// List of prepositions used in subject cleaning.
const PREPOSITIONS = ["in", "at", "on", "for", "to", "near"];

/**
 * Class for handling ML-based suggestions using intent and NER models.
 * @class
 */
class _MLSuggest {
  #modelEngines = {};

  async #initializeMLModel(options) {
    const engine_id = `${options.taskName}-${options.modelId}`;

    // uses cache if engine was used
    if (this.#modelEngines[engine_id]) {
      return this.#modelEngines[engine_id];
    }

    options.engineId = engine_id;
    const engine = await lazyModules.createEngine(options);
    // Cache the engine
    this.#modelEngines[engine_id] = engine; 
    return engine;
  }

  /**
   * Finds the intent of the query using the intent classification model.
   * @param {string} query - The user's input query.
   * @returns {Promise<string|null>} The predicted intent label or null if the model is not initialized.
   */
  async #findIntent(query) {
    const engineIntentClassifier = this.#modelEngines[`${INTENT_OPTIONS.taskName}-${INTENT_OPTIONS.modelId}`];
    
    if (!engineIntentClassifier) {
      console.error("Intent classifier model not initialized.");
      return null;
    }

    const request = { args: [query], options: {} };
    const res = await engineIntentClassifier.run(request);
    // Return the first label from the result  
    return res[0].label; 
  }

  /**
   * Finds named entities in the query using the NER model.
   * @param {string} query - The user's input query.
   * @returns {Promise<Object[]|null>} The NER results or null if the model is not initialized.
   */
  async #findNER(query) {
    const engineNER = this.#modelEngines[`${NER_OPTIONS.taskName}-${NER_OPTIONS.modelId}`];

    if (!engineNER) {
      console.error("NER model not initialized.");
      return null;
    }

    const request = { args: [query], options: {} };
    return engineNER.run(request);
  }

  /**
   * Combines location tokens detected by NER into a single string.
   * @param {Object[]} nerResult - The NER results.
   * @param {number} nerThreshold - The confidence threshold for including entities.
   * @returns {string|null} The combined location or null if no location is found.
   */
  async #combineLocations(nerResult, nerThreshold) {
    let locResult = [];

    for (let i=0; i < nerResult.length; i++) {
      const res = nerResult[i];
      if ((res.entity === "B-LOC" || res.entity === "I-LOC") && res.score > nerThreshold) {
        if (res.word.startsWith("##") && locResult.length) {
          // Append subword to last word
          locResult[locResult.length - 1] += res.word.slice(2); 
        } else {
          locResult.push(res.word);
        }
      }
    }

    return locResult.length ? locResult.join(" ") : null;
  }

  #findSubjectFromQuery(query, location) {
    // If location is null, return the entire query as the subject
    if (!location) {
      return query;
    }

    const subjectWithoutLocation = query
      .replace(location, "")
      .trim();
    return this.#cleanSubject(subjectWithoutLocation);
  }

  #cleanSubject(subject) {
    let end = PREPOSITIONS.find(p => subject === p || subject.endsWith(" " + p));
    if (end) {
      subject = subject.substring(0, subject.length - end.length).trimEnd();
    }
    return subject;
  }
  
  #sumObjectsByKey(...objs) {
    return objs.reduce((a, b) => {
      for (let k in b) {
        if (b.hasOwnProperty(k)) a[k] = (a[k] || 0) + b[k];
      }
      return a;
    }, {});
  }

  /**
   * Generates ML-based suggestions by finding intent, detecting entities, and combining locations.
   * @param {string} query - The user's input query.
   * @returns {Promise<Object|null>} The suggestion result including intent, location, and subject, or null if an error occurs.
   */
  async makeMLSuggestions(query) {
    // Return null if models are not initialized before
    if (!this.#modelEngines ||
        !this.#modelEngines[`${INTENT_OPTIONS.taskName}-${INTENT_OPTIONS.modelId}`] || 
        !this.#modelEngines[`${NER_OPTIONS.taskName}-${NER_OPTIONS.modelId}`]) {
      console.error("Models not initialized. Please call initialize() first.");
      return null;
    }

    let intentRes, nerResult;
    try {
      [intentRes, nerResult] = await Promise.all([
        this.#findIntent(query),
        this.#findNER(query),
      ]);
    } catch (error) {
      console.error("Error in model inference:", error);
      return null;
    }
    

    const locationResVal = await this.#combineLocations(
      nerResult,
      NER_THRESHOLD
    );
    const subjectRes = this.#findSubjectFromQuery(
      query,
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

    return finalRes;
  }

  /**
   * Shuts down all initialized engines.
   * @returns {Promise<void>}
   */
  async shutdown() {
    await Promise.all(
      Object.values(this.#modelEngines).map(async engine => {
        if (engine.terminate) {
          await engine.terminate();
        }
      })
    );
    // Clear the model engines after shutdown
    this.#modelEngines = {};
  }

  /**
   * Initializes the intent and NER models.
   * @returns {Promise<void>}
   */
  async initialize() {
    await Promise.all([
      this.#initializeMLModel(INTENT_OPTIONS),
      this.#initializeMLModel(NER_OPTIONS),
    ]);
  }
}

// Export the singleton instance
export var MLSuggest = new _MLSuggest();

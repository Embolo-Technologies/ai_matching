# Embolo Pharmacy Medicine Matching Engine

A fast, highly optimized pharmacy medicine matching engine designed to run efficiently on low-end hardware. It maps chemist inventory items against a wholesaler master catalog using multi-stage fuzzy matching (via `rapidfuzz`), prefix indexing, and custom phonetic/semantic rules for pharmaceutical names.

## Key Features

- **High Speed**: Matches hundreds of items per second using a pre-computed prefix index for wide-recall retrieval.
- **Accurate Matches**: Tailored text normalization rules for medicine names (handling strengths, units, brands, and formulations).
- **Multi-stage Scoring**: Combines rapidfuzz ratio methods with token overlaps and formulation compatibility checks.
- **Excel Batch Matching**: Scripts to match thousands of input items and export formatted Excel sheets grouped by confidence levels for easy human review.
- **Interactive UI**: Contains a standalone HTML search interface (`search_app.html`) connecting to a local Flask server for instant lookup testing.
- **Optional LLM Fallback**: Core integration ready for lightweight local LLM model execution (e.g., Qwen 1.7B) to handle extremely ambiguous matching cases.

## Repository Structure

- `qwen3_engine/`: Core Python package containing the matching server and business logic.
  - `fast_server.py`: Flask-based API server hosting the matching and search engine.
  - `searcher.py`: Prefix-indexed fuzzy searcher for O(1) candidate lookup.
  - `matcher.py`: Rules-based matching engine comparing brands, strengths, pack size, and formulation compatibility.
  - `pharma_data.py`: Compatibility checking logic (e.g., standardizing capsules, tablets, syrups).
  - `config.py` & `engine.py`: Infrastructure for loading local GGUF models.
- `search_app.html`: Built-in frontend dashboard to run search queries and view matches in real time.
- `setup.sh` / `setup.bat`: Scripts to initialize the virtual environment and install dependencies.
- `compile.sh` / `compile.bat`: Build wrappers to bundle or compile the fast search binary.

## Running the Server

1. **Initialize the Environment**:
   ```bash
   ./setup.sh
   ```

2. **Start the Fast Search Server**:
   ```bash
   ./run.sh
   ```
   This starts the Flask server at `http://127.0.0.1:5000`.

3. **Use the UI**:
   Open `search_app.html` in your web browser to start testing matches.

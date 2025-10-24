# SillyTavern Expression Classifier

SillyTavern Expression Classifier sorts batches of sprite or portrait images into facial-expression folders. Feed it a character sheet and it will:

- optionally resize the images into a common shape,
- send each frame to the configured classifier (default: Mistral's Pixtral-12B vision model),
- copy the file into an expression-specific subdirectory, and
- log predictions with confidence scores for auditing.

You can run the workflow from the command line or through the bundled Streamlit UI, so artists and engineers share the same pipeline.

---

## How It Works

1. **Mode selection** - `scripts/classify_batch.py` reads `classification_mode` from `config.yaml`. Choose `mistral`, `openai`, `google`, or `grok` for hosted APIs, or `local` to use a PyTorch checkpoint on disk.
2. **Image preparation** - if `data/processed/` exists it is preferred, otherwise files are read from `data/raw/`. Images are converted to RGB and resized to 224×224 for local inference; the API receives the raw bytes.
3. **Inference** -
   - *Remote mode*: the script base64-encodes the image and sends it, plus a label prompt, to the selected provider (Mistral, OpenAI, Google, or Grok). The JSON response is parsed for `label` and `confidence`.
   - *Local mode*: the PyTorch model is loaded onto CPU, torchvision transforms are applied, and softmax probabilities are computed to obtain the top label.
4. **Confidence flag** - if the score is below `confidence_threshold`, the entry is marked as low-confidence but still assigned to the predicted label for review.
5. **Sorted output** - predictions are copied into `<output>/<label>/` folders (default: `data/sorted/<label>/`) alongside a `classification_log.json` audit trail containing confidence flags, fallback reasons, per-image count snapshots, and raw API responses when applicable.
6. **Manual packaging** - once you are satisfied with the layout, run the standalone `scripts/postprocess_sorted.py` helper to rename files (`Label.###`), copy them into `data/finished/<total>-<date>-<time>/`, create an archive under `output_zips/`, and optionally clean up `data/sorted/`.
7. **UI option** - `ui/app.py` wraps the CLI so non-technical users can browse for folders and run the sorter from a browser.

---

## Project Layout

```text
.
├── config.default.yaml       # Template configuration copied to config.yaml on first run
├── data/
│   ├── raw/                  # Drop unsorted sprites here
│   ├── processed/            # Optional resized copy of raw assets
│   ├── sorted/               # Classification outputs awaiting review/post-processing
│   └── finished/             # Timestamped exports produced by postprocess_sorted.py
├── models/
│   ├── emotion_classifier.pth  # Optional local PyTorch weights
│   └── README_MODEL.md         # Notes about the model and training metadata
├── scripts/
│   ├── preprocess.py         # Minimal 224×224 resize pipeline
│   ├── classify_batch.py     # Main CLI entry point
│   ├── postprocess_sorted.py # Standalone renaming/combine workflow
│   └── utils.py              # Helpers for I/O and directory management
├── ui/
│   └── app.py                # Streamlit front-end
├── setup.sh                  # Bash bootstrapper for the Python environment
├── setup_config.sh           # Optional overrides for setup.sh (e.g., PyTorch index)
└── requirements.txt          # Python dependencies
```

---

## Requirements

- Python 3.10 or newer
- pip plus `venv`/virtualenv
- A Mistral API key (for the default API mode)
- PyTorch & torchvision (already listed in `requirements.txt`) if you plan to run the optional local mode

The provided `setup.sh` installs PyTorch from the CUDA 13.0 wheel index. Override the URL in `setup_config.sh`—for example, to `https://download.pytorch.org/whl/cpu`—if you need a different build.

---

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/<you>/sillytavern-expression-classifier.git
   cd sillytavern-expression-classifier
   ```

2. **Create the virtual environment**
   - POSIX / WSL:

     ```bash
     ./setup.sh
     ```

   - Windows PowerShell (manual sequence):

     ```powershell
     py -3 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install --upgrade pip
     pip install -r requirements.txt
     ```

3. **Activate the environment whenever you work on the project**
   - POSIX / WSL: `source .venv/bin/activate`
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`

4. **Create and configure `config.yaml`**
   - On first run the CLI copies `config.default.yaml` to `config.yaml`; you can also copy it manually.  
   - Choose a provider by setting `classification_mode` to `mistral`, `openai`, `google`, `grok`, or `local`, then supply the matching API key (`MISTRAL_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`) or populate the corresponding key in `config.yaml`.  
   - Adjust `emotion_classes`, `confidence_threshold`, `api_prompt_template`, and `bias_prompt_template` to match your use case.

5. **Optional: prepare local weights**
   - Switch `classification_mode` to `"local"` if you want to bypass the API.
   - Place your PyTorch checkpoint at `models/emotion_classifier.pth` (or override `model_path`).
   - Document the training recipe in `models/README_MODEL.md`.

6. **Prepare input assets**
   - Copy the images you want to classify into `data/raw`. If you keep multiple character sets, run the classifier on each folder separately.

---

## Configuration Reference

All runtime settings live in `config.yaml`, which is generated from `config.default.yaml` if it does not exist. The generated file is ignored by git so you can store API keys safely.

Key entries in `config.yaml`:

- `classification_mode`: set to `"mistral"`, `"openai"`, `"google"`, `"grok"`, or `"local"`; legacy value `"api"` aliases to `"mistral"`.  
- `mistral_api_key` / `openai_api_key` / `google_api_key` / `xai_api_key`: credentials for the selected provider. You can also rely on the environment variables `MISTRAL_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `XAI_API_KEY`.  
- `mistral_model`, `openai_model`, `google_model`, `xai_model`: default vision-capable model IDs for each provider.  
- `mistral_endpoint`, `openai_endpoint`, `google_endpoint`, `xai_endpoint`: override endpoints if you need to hit a proxy or different API host.  
- `api_prompt_template`: Multi-line string that becomes the base prompt sent to the vision model; `{labels}` is replaced with the comma-separated list from `emotion_classes`.  
- `bias_prompt_template`: Optional snippet appended to the prompt each request. `{counts_json}` is replaced with a JSON object of current label usage so you can bias toward underrepresented emotions.
- `model_path`: Path to the `.pth` file for local mode.
- `input_dir`, `processed_dir`, `output_dir`: Source/processed/output directories.
- `emotion_classes`: Ordered label list that aligns with whichever classifier you use.
- `confidence_threshold`: Minimum confidence considered reliable. Predictions below this value keep their label but are flagged as `below_threshold` in the log.

All of the path-like values can be overridden at runtime with CLI flags (`--input-dir`, `--processed-dir`, `--output-dir`, `--model-path`, `--config`). Pass `--mode mistral|openai|google|grok|local` to override `classification_mode` without editing the file.

---

## Usage

### (Optional) Preprocess sprites

Normalize image size and RGB format before classifying:

```bash
python scripts/preprocess.py
```

The script copies every supported file (`.png`, `.jpg`, `.jpeg`, `.webp`) from `data/raw` into `data/processed`, resized to 224×224.

### Batch classification (CLI)

```bash
python scripts/classify_batch.py
```

Common overrides:

```bash
python scripts/classify_batch.py \
  --config path/to/config.yaml \
  --mode mistral \
  --input-dir path/to/images \
  --processed-dir path/to/preprocessed \
  --output-dir path/to/output
```

- Remote mode expects a valid API key for the chosen provider (Mistral/OpenAI/Google/xAI) in the config file or environment.  
- Local mode additionally needs `--model-path` (or the `model_path` entry in `config.yaml`).  
- Use `--mode openai`, `--mode google`, or `--mode grok` to target alternative providers without editing the config file.  
- Outputs include sorted images in `<output>/<label>/` and a `classification_log.json` file capturing filename, predicted label, assigned label, confidence, `below_threshold` status, classifier/provider metadata, raw model output, any fallback reason, the counts JSON supplied to the prompt, and the exact text sent for that image. Run `python scripts/postprocess_sorted.py` when you're ready to package the results.

`classification_log.json` is a JSON object with run metadata (timestamps, mode, directories, prompt templates, final counts), a `post_process` section noting the pending status and providing pointers to `data/sorted` / `data/finished`, and an `entries` array containing the per-image records described above. The post-processing helper works on the filesystem outputs but does not rewrite the log.

### Rename, combine, and package results

After you are satisfied with the sorted folders, run:

```bash
python scripts/postprocess_sorted.py
```

The helper will:

- rename files inside each label directory to `Label.###` (with de-duplication if needed),
- copy the renamed files into a new timestamped directory under `data/finished/`,
- prompt for a zip name and create `<name>.zip` inside `output_zips/`, and
- offer to delete the remaining contents of `data/sorted/` (with a reminder that the finished copy is preserved).

Use `--output-dir` if you classified into a non-default location, e.g. `python scripts/postprocess_sorted.py --output-dir path/to/output`.

### Streamlit UI

```bash
streamlit run ui/app.py
```

Provide the input and output directories, then click **Run Classification**. Remember to set the appropriate provider API key (for example `MISTRAL_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `XAI_API_KEY`) before launching the UI; the app simply invokes the CLI under the hood.

---

## Remote Classification Notes

- Pixtral-12B (`pixtral-12b-latest`) remains the default, but you can switch to OpenAI (`openai_model`, e.g., `gpt-4o-mini`), Google (`google_model`, e.g., `gemini-1.5-flash`), or xAI's Grok (`xai_model`, e.g., `grok-4-fast-reasoning`) by updating `classification_mode`.  
- The prompt enforces a JSON-only reply. If the response drifts from the expected structure, the script falls back to the least-used label, records the raw text, and notes the issue in `fallback_reason`.  
- Each provider has overrideable endpoints in `config.yaml` so you can point at gateways or future API revisions without touching the code.  
- When using Grok, ensure your key has access to multimodal endpoints; the tool sends images inline using base64 `image_url` payloads compatible with the Grok chat completions API.

---

## Local Mode Notes

- A trained PyTorch checkpoint is not included. Supply your own weights at `models/emotion_classifier.pth` and ensure the classifier head order matches `emotion_classes`.
- The preprocessing assumes 224×224 RGB inputs normalized like ImageNet. If your model expects different transforms, update `scripts/utils.py` and the transform pipeline inside `scripts/classify_batch.py`.
- For GPU inference, modify `run_local_classification` to move the model and tensors to CUDA.

---

## Troubleshooting

- **Missing API key** - set the provider-specific key (`mistral_api_key`, `openai_api_key`, or `google_api_key`) in `config.yaml` or export `MISTRAL_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY`.
- **HTTP errors** - check network connectivity and your provider quota; failed requests are logged and the image is reassigned to the least-used label with a `fallback_reason`.
- **Unsupported image format** - convert files outside of `.png`, `.jpg`, `.jpeg`, or `.webp`.
- **Excess low-confidence results** - lower `confidence_threshold`, adjust the prompt, or fine-tune a dedicated local model.
- **Local mode model mismatch** - confirm the checkpoint's classifier head matches the number and order of labels in `emotion_classes`.

Happy sorting!

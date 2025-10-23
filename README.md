# SillyTavern Expression Classifier

SillyTavern Expression Classifier sorts batches of sprite or portrait images into facial-expression folders. Feed it a character sheet and it will:

- optionally resize the images into a common shape,
- send each frame to the configured classifier (default: Mistral's Pixtral-12B vision model),
- copy the file into an expression-specific subdirectory, and
- log predictions with confidence scores for auditing.

You can run the workflow from the command line or through the bundled Streamlit UI, so artists and engineers share the same pipeline.

---

## How It Works

1. **Mode selection** - `scripts/classify_batch.py` reads `classification_mode` from `config.yaml`. `api` (default) calls Mistral's Pixtral model; `local` loads a PyTorch checkpoint from disk.  
2. **Image preparation** - if `data/processed/` exists it is preferred, otherwise files are read from `data/raw/`. Images are converted to RGB and resized to 224×224 for local inference; the API receives the raw bytes.  
3. **Inference** -  
   - *API mode*: the script base64-encodes the image and sends it, plus a label prompt, to the Mistral Chat Completions endpoint. The JSON response is parsed for `label` and `confidence`.  
   - *Local mode*: the PyTorch model is loaded onto CPU, torchvision transforms are applied, and softmax probabilities are computed to obtain the top label.  
4. **Thresholding** - if the confidence score falls below `confidence_threshold`, the image is filed under `uncertain` for manual review.  
5. **Output** - files are copied into `<output>/<label>/` folders and a `classification_log.json` audit trail is written alongside the results (including raw API responses when applicable).  
6. **UI option** - `ui/app.py` wraps the CLI so non-technical users can browse for folders and run the sorter from a browser.

---

## Project Layout

```text
.
├── config.yaml               # Paths, classifier mode, credentials, labels, thresholds
├── data/
│   ├── raw/                  # Drop unsorted sprites here
│   ├── processed/            # Optional resized copy of raw assets
│   └── sorted/               # Classification outputs (one folder per label + uncertain)
├── models/
│   ├── emotion_classifier.pth  # Optional local PyTorch weights
│   └── README_MODEL.md         # Notes about the model and training metadata
├── scripts/
│   ├── preprocess.py         # Minimal 224×224 resize pipeline
│   ├── classify_batch.py     # Main CLI entry point
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

4. **Configure classification**
   - Open `config.yaml` and set `mistral_api_key` (or export `MISTRAL_API_KEY`) for API mode.  
   - Adjust `emotion_classes` and `confidence_threshold` to match your use case.

5. **Optional: prepare local weights**
   - Switch `classification_mode` to `"local"` if you want to bypass the API.  
   - Place your PyTorch checkpoint at `models/emotion_classifier.pth` (or override `model_path`).  
   - Document the training recipe in `models/README_MODEL.md`.

6. **Prepare input assets**
   - Copy the images you want to classify into `data/raw`. If you keep multiple character sets, run the classifier on each folder separately.

---

## Configuration Reference

Key entries in `config.yaml`:

- `classification_mode`: `"api"` (default) or `"local"`.  
- `mistral_api_key`: API key used when `classification_mode` is `api`. You can leave this blank and rely on the `MISTRAL_API_KEY` environment variable.  
- `mistral_model`: Mistral model name; defaults to `"pixtral-12b-latest"`.  
- `mistral_endpoint`: Chat completions endpoint; defaults to `https://api.mistral.ai/v1/chat/completions`.  
- `model_path`: Path to the `.pth` file for local mode.  
- `input_dir`, `processed_dir`, `output_dir`: Source/processed/output directories.  
- `emotion_classes`: Ordered label list that aligns with whichever classifier you use.  
- `confidence_threshold`: Minimum confidence required to accept the predicted label; lower scores are moved to `uncertain`.

All of the path-like values can be overridden at runtime with CLI flags (`--input-dir`, `--processed-dir`, `--output-dir`, `--model-path`, `--config`). Pass `--mode api|local` to override `classification_mode` without editing the file.

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
  --mode api \
  --input-dir path/to/images \
  --processed-dir path/to/preprocessed \
  --output-dir path/to/output
```

- API mode expects a valid key in `mistral_api_key` or the `MISTRAL_API_KEY` environment variable.  
- Local mode additionally needs `--model-path` (or the `model_path` entry in `config.yaml`).  
- Outputs include sorted images in `<output>/<label>/`, an `uncertain/` folder for low-confidence cases, and `classification_log.json` with filename, predicted label, assigned label, confidence, classifier mode, and the raw API text (if applicable).

### Streamlit UI

```bash
streamlit run ui/app.py
```

Provide the input and output directories, then click **Run Classification**. Remember to set `MISTRAL_API_KEY` (or update `config.yaml`) before launching the UI if you plan to use API mode; the app simply invokes the CLI under the hood.

---

## API Classification Notes

- Pixtral-12B (`pixtral-12b-latest`) is the default because it offers the most consistent fine-grained expression recognition among Mistral's released models.  
- The prompt enforces a JSON-only reply. If the response drifts from the expected structure, the script falls back to `uncertain` and records the raw text in the log for debugging.  
- You can change `mistral_model` or `mistral_endpoint` in `config.yaml` to target other Mistral vision-capable endpoints without modifying the code.

---

## Local Mode Notes

- A trained PyTorch checkpoint is not included. Supply your own weights at `models/emotion_classifier.pth` and ensure the classifier head order matches `emotion_classes`.  
- The preprocessing assumes 224×224 RGB inputs normalized like ImageNet. If your model expects different transforms, update `scripts/utils.py` and the transform pipeline inside `scripts/classify_batch.py`.  
- For GPU inference, modify `run_local_classification` to move the model and tensors to CUDA.

---

## Troubleshooting

- **Missing API key** - set `mistral_api_key` in `config.yaml` or export `MISTRAL_API_KEY`.  
- **HTTP errors** - check network connectivity and your Mistral quota; failed requests are logged and the image is marked `uncertain`.  
- **Unsupported image format** - convert files outside of `.png`, `.jpg`, `.jpeg`, or `.webp`.  
- **Too many uncertain results** - lower `confidence_threshold`, adjust the prompt, or fine-tune a dedicated local model.  
- **Local mode model mismatch** - confirm the checkpoint's classifier head matches the number and order of labels in `emotion_classes`.

Happy sorting!

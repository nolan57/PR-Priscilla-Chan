# Pro Singer Separator: Initialization and Model-Loading Logic

This document explains, in detail, how `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py` initializes and how it loads models.

## 1. Startup Sequence (Top-Level Script Init)

The script executes initialization in this order:

1. Imports standard library modules (`gc`, `os`, `subprocess`, `sys`, etc.).
2. Sets environment variables **before heavy ML imports**.
3. Imports ML/audio/UI dependencies (`librosa`, `numpy`, `torch`, `PyQt6`, etc.).
4. Determines compute device (`cuda` -> `mps` -> `cpu`).
5. Discovers `ultimatevocalremovergui` root and inserts it into `sys.path`.
6. Attempts optional imports (Demucs, SpeechBrain).
7. Defines lazy Whisper import gate.
8. Creates path/config dataclasses and helper classes.
9. Builds engine/UI classes.
10. In `__main__`, starts Qt app and shows the main window.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:13`
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:19`
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:57`
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:71`

## 2. Environment Variables and Why They Are Early

The script sets these at import time:

- `PYTORCH_ENABLE_MPS_FALLBACK=1`
  - Allows PyTorch operations to fallback when MPS lacks kernels.
- `KMP_DUPLICATE_LIB_OK=TRUE`
  - Workaround for duplicate OpenMP runtime abort on macOS.
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
  - Prevents runtime downloads for HuggingFace-based components.

They are set before importing `torch`/`librosa` to avoid initialization-order issues.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:13`

## 3. Device Detection

`get_best_device()` picks runtime device in priority order:

1. `cuda` if available.
2. `mps` if available and built.
3. otherwise `cpu`.

This global `DEVICE` is reused for model loading and inference.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:57`
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:65`

## 4. UVR Root Discovery and Python Path Injection

`_find_uvr_root()` tries:

1. Preferred path: `<project_root>/ultimatevocalremovergui`
2. Recursive fallback: search under project for folder named `ultimatevocalremovergui` containing `separate.py`

If found, that path is prepended to `sys.path`.

Why it matters:
- Enables importing UVR-bundled Python modules (e.g., Demucs package copy).
- Ensures the script can execute `ultimatevocalremovergui/separate.py`.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:71`
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:82`

## 5. Optional Dependency Probing

The script does not hard-fail when optional packages are missing.

### 5.1 Demucs

- Tries `from demucs.apply import apply_model` and `from demucs.pretrained import get_model`.
- Sets:
  - `DEMUCS_AVAILABLE=True` on success
  - `DEMUCS_AVAILABLE=False` on failure

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:87`

### 5.2 SpeechBrain (speaker embedding)

- Tries `from speechbrain.inference import SpeakerRecognition`.
- Sets:
  - `SPEECHBRAIN_AVAILABLE=True` on success
  - `False` on failure

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:102`

### 5.3 Whisper

Whisper is explicitly lazy and gated:

- `try_import_whisper()` returns immediately unless env var is set:
  - `PRO_SINGER_ENABLE_WHISPER=1`
- If enabled, it imports `whisper` dynamically and sets `WHISPER_AVAILABLE=True`.
- On any import failure, it remains unavailable and does not crash startup.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:118`

## 6. Path and Settings Objects

Two dataclasses structure app state:

- `AppPaths`
  - `project_root`, `uvr_root`, `model_root`, `mdx_models`, `vr_models`, `demucs_models`
- `SeparationSettings`
  - `model_name`, threshold, denoise/use_uvr flags, aggression, sample rate, quality

`resolve_paths()` maps `ultimatevocalremovergui/models` into these fields.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:136`
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:157`

## 7. ModelLocator: How UI Model Choices Are Built and Resolved

`ModelLocator` encapsulates model discovery.

### 7.1 `list_primary_models()`

- Reads available models from:
  - `models/MDX_Net_Models` (`*.onnx`, `*.ckpt`)
  - `models/VR_Models` (`*.pth`)
- Prepends a preferred order list for common UVR Inst models.
- De-duplicates while preserving order.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:184`

### 7.2 `resolve(model_name)`

Resolution order:

1. direct lookup in MDX/VR/Demucs/model root
2. fallback trying extension variants: `""`, `.onnx`, `.ckpt`, `.pth`, `.th`, `.yaml`

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:210`

## 8. VocalIsolationEngine Initialization

`VocalIsolationEngine.__init__()`:

1. stores paths and creates `ModelLocator`
2. initializes model handles (`demucs_model`, `speaker_model`, `whisper_model`)
3. creates process lock and active-process pointer for cancellation
4. calls `_load_optional_models()`

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:233`

### 8.1 `_load_optional_models()` details

#### Demucs load

- First try: `get_model("htdemucs_ft", repo=None)`
- Fallback: `get_model("htdemucs_ft")`
- On both fail: leaves `self.demucs_model=None`

#### Speaker model load

Only attempts if SpeechBrain import succeeded.

Searches `embedding_model` in this order:

1. `<project_root>/models/embedding_model`
2. `<script_dir>/models/embedding_model`
3. `<uvr_model_root>/embedding_model`

If found, loads with `SpeakerRecognition.from_hparams(..., run_opts={"device": DEVICE})`.

#### Whisper model load

- Calls `try_import_whisper()` first (env-gated).
- If true, attempts `whisper.load_model("small")`.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:245`

## 9. Separation Runtime Pipeline (Where Model Loading Is Used)

Main runtime function: `separate_target_voice()`.

Steps:

1. Load input wave.
2. Separation stage:
   - if `use_uvr`: run UVR CLI (`separate.py`)
   - fallback to Demucs/spectral
3. Build target embeddings from references.
4. Build frame-wise speaker mask.
5. Apply post-processing and normalization.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:287`

### 9.1 UVR CLI execution path

`_separate_with_uvr_cli()`:

1. Resolves selected model via `ModelLocator.resolve()`.
2. Creates temp input/output directories.
3. Writes input audio to temp `source.wav`.
4. Executes:
   - `sys.executable <uvr_root>/separate.py --input_folder ... --output_folder ... --model <model_name> --vocals_only True`
   - adds `--no_cuda` when `DEVICE==cpu`
5. Captures stdout, checks return code.
6. Picks newest vocals-like output WAV.
7. Loads it and returns mono vocals.

Key point:
- Uses **the same Python interpreter** as the GUI (`sys.executable`).

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:321`
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:342`

### 9.2 Demucs/spectral fallback

`_demucs_or_spectral()`:

- If Demucs model is loaded, performs Demucs inference and takes source index 3 (vocals).
- Otherwise applies spectral vocal enhancement fallback.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:392`

## 10. App Initialization and UI Wiring

`ProSingerSeparatorApp.__init__()`:

1. `self.paths = resolve_paths()`
2. `self.engine = VocalIsolationEngine(self.paths)` (this triggers optional model loads)
3. Initializes runtime state (audio buffers, worker, playback)
4. Creates UI and updates status panel

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:644`

### 10.1 Settings tab model dropdown

- Built from `self.engine.model_locator.list_primary_models()`.
- If empty, inserts default fallback entry.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:824`

### 10.2 Status panel information

`_update_model_status()` displays:

- active Python path (`sys.executable`)
- selected device
- UVR root and models root
- Demucs availability/load state
- Speaker model load state
- Whisper load state

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:869`

## 11. Why Speaker/Whisper Show "Unavailable"

Given current logic, status becomes unavailable if:

### Speaker unavailable when:

1. `speechbrain` import fails, or
2. no `embedding_model` folder exists in any checked location, or
3. `from_hparams` load fails (model files invalid/incomplete/device mismatch)

### Whisper unavailable when:

1. `PRO_SINGER_ENABLE_WHISPER` is not set to `1`, or
2. `whisper` package is not installed, or
3. `whisper.load_model("small")` fails

## 12. Concurrency and Cancellation at Init/Runtime Boundary

- `SeparationWorker` runs separation in a `QThread` to keep UI responsive.
- `VocalIsolationEngine` tracks active UVR subprocess under a lock.
- `cancel_active_job()` terminates subprocess safely when user cancels/closes.

Relevant code:
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:278`
- `/Users/lpcw/Documents/PR-Priscilla-Chan/audio-edit/pro_singer_separator.py:534`

## 13. Practical Verification Checklist

When debugging init/model-load issues, verify in this order:

1. Status panel `Python:` path is the expected interpreter.
2. `UVR root` points to your repo’s `ultimatevocalremovergui`.
3. `UVR models` points to `ultimatevocalremovergui/models`.
4. Selected model exists on disk and is resolvable by `ModelLocator`.
5. For speaker model: check `embedding_model` folder presence.
6. For Whisper: export `PRO_SINGER_ENABLE_WHISPER=1` before launch.

---

If you want, I can also add a short "Initialization Debug" button in the app that dumps all path checks and model-probe results to a dialog/file.

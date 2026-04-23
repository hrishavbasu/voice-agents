# Hindi Audio Transcription & Translation — POC

A proof-of-concept Python script that transcribes a Hindi/Hinglish audio file using
**Google Cloud Speech-to-Text v2** and translates the output to English using
**Google Cloud Translation API**.

---

## What It Does

1. Takes an audio file (`output.aac`) as input
2. Converts it to FLAC format using `ffmpeg` (strips unsupported encoding)
3. Trims silence/ringing from the beginning (first 30 seconds in our case)
4. Uploads the processed audio to **Google Cloud Storage (GCS)**
5. Sends it to **Google Speech-to-Text v2** (`batch_recognize`) for transcription
6. Pipes the transcript through **Google Translate** to produce an English translation
7. Prints both the original transcript and English translation to the terminal

---

## Pipeline

```
output.aac
    │
    ▼
ffmpeg (trim 30s ringing + convert to FLAC @ 16kHz mono)
    │
    ▼
output_trimmed.flac
    │
    ▼
Google Cloud Storage (auto-created bucket)
    │
    ▼
Google Speech-to-Text v2  ── language: hi-IN + en-IN (Hinglish)
    │                         model: long (no duration limit)
    ▼
Hindi/Hinglish Transcript
    │
    ▼
Google Cloud Translation API
    │
    ▼
English Translation
```

---

## Tech Stack

| Component | Service | Notes |
|-----------|---------|-------|
| Speech-to-Text | Google Cloud Speech-to-Text v2 | `batch_recognize`, supports >60s audio |
| Translation | Google Cloud Translation API v2 | Basic edition, hi → en |
| Storage | Google Cloud Storage | Auto-created bucket for audio upload |
| Audio processing | ffmpeg | AAC → FLAC conversion, trimming |
| Auth | GCP Application Default Credentials (ADC) | No service account key needed |

---

## Prerequisites

### 1. GCP Setup (one-time)

```bash
# Install gcloud CLI
brew install --cask google-cloud-sdk

# Authenticate
gcloud auth login
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable speech.googleapis.com
gcloud services enable translate.googleapis.com
gcloud services enable storage.googleapis.com
```

### 2. Install ffmpeg

```bash
brew install ffmpeg
```

### 3. Python Dependencies

```bash
cd /path/to/project
python -m venv .venv
source .venv/bin/activate
pip install google-cloud-speech==2.37.0 google-cloud-storage google-cloud-translate
```

---

## Audio Pre-processing

The input file `output.aac` has ~30 seconds of ringing/silence before speech begins.
Run these ffmpeg commands once before running the script:

```bash
# Step 1: Convert AAC → FLAC (Google STT does not support AAC directly)
ffmpeg -i output.aac -ar 16000 -ac 1 output.flac

# Step 2: Trim the first 30 seconds of ringing
ffmpeg -i output.aac -ss 30 -ar 16000 -ac 1 output_trimmed.flac
```

> If the ringing duration differs, adjust `-ss 30` accordingly.
> Use `ffplay output.aac` to listen and find the exact start of speech.

---

## Running the Script

```bash
source .venv/bin/activate
export GOOGLE_CLOUD_PROJECT=your-actual-project-id

python test_google_stt.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | `project-e264ab5e-...` | GCP project ID |
| `AUDIO_FILE` | `output_trimmed.flac` | Path to audio file |
| `GCS_BUCKET` | `{project-id}-stt-tmp` | GCS bucket name (auto-created) |

### Expected Output

```
Using existing bucket: your-project-stt-tmp
Uploading output_trimmed.flac to gs://your-project-stt-tmp/audio/abc123_output_trimmed.flac ...
Upload complete.
Transcribing (this may take a minute)...

=== Transcript ===
हाँ, मैं आपकी कैसे मदद कर सकता हूँ...
Confidence: 0.92

=== English Translation ===
Yes, how can I help you...
```

---

## Cost Estimate

| Service | Usage | Approx. Cost |
|---------|-------|-------------|
| Speech-to-Text (`long` model) | ~4.5 min audio = 18 × 15s chunks | ~$0.07 per run |
| Cloud Translation | ~500 chars transcript | ~$0.01 per run |
| Cloud Storage | <10MB upload | negligible |
| **Total per run** | | **~$0.08** |

> Monitor costs: GCP Console → Billing → Reports → filter by API.
> Set a budget alert: Billing → Budgets & Alerts → Create Budget.

---

## Known Limitations

- **Sync `recognize` API has a 60s limit** — this script uses `batch_recognize` (long-running operation) to handle full-length calls
- **AAC not supported** by Google STT auto-detect — must convert to FLAC via ffmpeg
- **Leading silence/ringing must be trimmed manually** — the 30s offset is hardcoded in the ffmpeg command
- **Translation accuracy** may vary for heavy Hinglish (code-switching) — works best when transcript is predominantly Hindi

---

## File Structure

```
.
├── test_google_stt.py       # Main script
├── output.aac               # Original audio file (input)
├── output.flac              # AAC → FLAC conversion
├── output_trimmed.flac      # Trimmed FLAC (30s ringing removed) — used by script
└── README_STT.md            # This file
```

---

## Next Steps / Improvements

- [ ] Auto-detect silence boundary instead of hardcoded 30s trim (using `ffmpeg silencedetect`)
- [ ] Speaker diarization (identify who is speaking — agent vs customer)
- [ ] Export transcript + translation to a structured format (JSON/CSV)
- [ ] Integrate into the main voice pipeline for real-time call transcription
- [ ] Swap Google Translate for an LLM-based translation for better Hinglish handling

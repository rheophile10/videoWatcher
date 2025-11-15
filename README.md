# governmentWatcher

## What it does (so far):
simple sqlite interface (pretty hacky)


## Goal
Watches official websites of legislatures and federal bodies.
Downloads every new video as soon as it’s posted.
Transcribes the audio.
Keeps a permanent, searchable record.

## Why:

perpetual scrutiny of utterances of elected officials

# requirements
you need ffmpeg and it needs to be in PATH
playwright install-deps
playwright install
you need a hugging face token for speaker diarization
python -c "from huggingface_hub import login; login()"
#  Required models (all run in <14 GB VRAM):
| Task                | Model                                      | VRAM   | Size on disk | Notes / Quality (2025)                         |
|---------------------|--------------------------------------------|--------|--------------|------------------------------------------------|
| **Diarization**     | `pyannote/speaker-diarization-3.1`         | ~10 GB | ~1.6 GB      | Current open-source SOTA for clean YouTube/podcasts/meetings |
| **Transcription**   | `openai/whisper-large-v3-turbo` (via faster-whisper) | ~7 GB  | ~1.6 GB      | Best open transcription model (word-level timestamps) |
| **Embedding**       | `BAAI/bge-large-en-v1.5`                   | ~5 GB  | ~1.3 GB      | #1 open retrieval embedding model in 2025     |
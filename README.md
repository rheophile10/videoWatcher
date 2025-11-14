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
you need a hugging face token for speaker diarization
python -c "from huggingface_hub import login; login()"
#  Required models (all run in <14 GB VRAM):
    • diar_msdd_telephonic (NVIDIA NeMo MSDD)   → speaker turns
    • openai/whisper-large-v3-turbo             → transcription + word timestamps
    • BAAI/bge-large-en-v1.5                    → best open embedding model (768 dim)
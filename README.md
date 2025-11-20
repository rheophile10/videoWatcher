# governmentWatcher

## What it does (so far):
[x] simple sqlite interface (pretty hacky)
[x] downloads vids from cpac
[x] transcribes and diarizes (but doesn't align words) of video audio
[ ] makes briefs about what was talked about <- doesn't do this yet

## Goal
Watches official websites of legislatures and federal bodies.
Downloads every new video as soon as it’s posted.
Transcribes the audio.
Keeps a permanent, searchable record.

## Why:

perpetual scrutiny of utterances of elected officials

# requirements
playwright install-deps
playwright install
you need a hugging face token for speaker diarization
python -c "from huggingface_hub import login; login()"


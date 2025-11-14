"""Full pipeline: Diarization → Transcription → Chunking → Embedding."""

from pathlib import Path
from typing import Dict, Generator, List, Tuple, Any, Literal

import torch
import numpy as np
from faster_whisper import WhisperModel
from nemo.collections.asr.models import ClusteringDiarizer
from sentence_transformers import SentenceTransformer


def prepare_models() -> Tuple[WhisperModel, ClusteringDiarizer, SentenceTransformer]:
    """Load models into memory."""
    DEVICE: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    whisper = WhisperModel("large-v3-turbo", device=DEVICE, compute_type="float16")
    diarizer = ClusteringDiarizer.from_pretrained("diar_msdd_telephonic")
    diarizer = diarizer.to(DEVICE)
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)
    return whisper, diarizer, embedder


def diarize_video(
    diarizer: ClusteringDiarizer, video_path: str
) -> Tuple[List[Tuple[int, float, float]], Dict[str, int]]:
    """Diarize a video file and return the diarizer object."""
    diarizer.audio_file = video_path
    diarizer.diarize()
    rttm_lines = (
        diarizer.rttm_lines if hasattr(diarizer, "rttm_lines") else diarizer._last_rttm
    )
    turn_inserts, speaker_map = _parse_rttm_lines(rttm_lines)
    return turn_inserts, speaker_map


def _parse_rttm_lines(
    rttm_lines: List[str],
) -> Tuple[List[Tuple[int, float, float]], Dict[str, int]]:
    """Parse RTTM lines into turn inserts and speaker map."""
    speaker_map: Dict[str, int] = {}
    turn_inserts = []
    for line in rttm_lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        speaker_label = parts[7]
        start = float(parts[3])
        duration = float(parts[4])
        end = start + duration

        if speaker_label not in speaker_map:
            speaker_map[speaker_label] = len(speaker_map) + 1
        turn_inserts.append((speaker_map[speaker_label], start, end))
    return turn_inserts, speaker_map


def _transcribe_video(
    whisper: WhisperModel, video_path: Path
) -> Generator[Any, None, None]:
    segments, _ = whisper.transcribe(
        str(video_path),
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )
    for segment in segments:
        yield segment


def _chunk_text(
    segments_gen: Generator[Any, None, None],
    turn_inserts: List[Tuple[int, float, float]],
    speaker_map: Dict[str, int],
    video_id: int,
) -> Generator[Dict[str, Any], None, None]:
    current_para_text = ""
    current_para_start = None
    current_para_speaker = None

    for seg in segments_gen:
        text = seg.text.strip()
        if not text:
            continue

        mid = (seg.start + seg.end) / 2
        speaker_id = None
        for turn in turn_inserts:
            if turn[1] <= mid < turn[2]:  # Assuming turn is (speaker_id, start, end)
                speaker_id = turn[0]
                break
        if speaker_id is None:
            speaker_id = list(speaker_map.values())[0]

        # Paragraph merging (same speaker, <8s gap)
        if (
            current_para_speaker == speaker_id
            and current_para_start is not None
            and seg.start - current_para_start < 8.0
        ):
            current_para_text += " " + text
            current_para_end = seg.end
        else:
            if current_para_text:
                chunk_record = {
                    "video_id": video_id,
                    "speaker_id": current_para_speaker,
                    "start_sec": current_para_start,
                    "end_sec": current_para_end,
                    "layer": "transcript",
                    "chunk_type": "turn",
                    "text": current_para_text,
                }
                yield chunk_record
            current_para_text = text
            current_para_start = seg.start
            current_para_end = seg.end
            current_para_speaker = speaker_id

    # Flush last paragraph
    if current_para_text:
        chunk_record = {
            "video_id": video_id,
            "speaker_id": current_para_speaker,
            "start_sec": current_para_start,
            "end_sec": current_para_end,
            "layer": "transcript",
            "chunk_type": "turn",
            "text": current_para_text,
        }
        yield chunk_record


def _make_chunk_embedding(
    embedder: SentenceTransformer, chunk: Dict[str, Any]
) -> Dict[str, Any]:
    emb = embedder.encode(chunk["text"], normalize_embeddings=True)
    emb_bytes = np.array(emb, dtype=np.float32).tobytes()
    embedding_record = {
        "embedding": emb_bytes,
        "text": chunk["text"],
        "video_id": chunk["video_id"],
        "layer": chunk["layer"],
        "chunk_type": chunk["chunk_type"],
    }
    return embedding_record


def process_transcription_and_embed(
    turn_inserts: List[Tuple[int, float, float]],
    speaker_map: Dict[str, int],
    whisper: WhisperModel,
    embedder: SentenceTransformer,
    video_path: Path,
    video_id: int,
) -> Generator[Tuple[Dict[str, Any], Dict[str, Any]], None, None]:
    """
    Orchestrates transcription, chunking, and embedding generation.
    Yields tuples of (chunk_record, embedding_record) for each chunk.
    """
    for chunk in _chunk_text(
        _transcribe_video(whisper, video_path), turn_inserts, speaker_map, video_id
    ):
        # Generate embedding for the chunk
        embedding_record = _make_chunk_embedding(embedder, chunk)
        yield chunk, embedding_record


def orchestrate_transcription_and_embedding(
    video_path: Path,
    video_id: int,
) -> Generator[Tuple[Dict[str, Any], Dict[str, Any]], None, None]:
    """
    Full pipeline: Diarization → Transcription → Chunking → Embedding.
    Yields tuples of (chunk_record, embedding_record) for each chunk.
    """
    diarizer, whisper, embedder = prepare_models()

    turn_inserts, speaker_map = diarize_video(diarizer, str(video_path))
    for chunk, embedding_record in process_transcription_and_embed(
        turn_inserts,
        speaker_map,
        whisper,
        embedder,
        video_path,
        video_id,
    ):
        yield chunk, embedding_record


__all__ = ["orchestrate_transcription_and_embedding"]

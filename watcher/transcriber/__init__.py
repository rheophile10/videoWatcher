"""
Transcription and diarization module using Whisper and PyAnnote.
"""

from pathlib import Path
from typing import Generator, Tuple, Dict, List, Any, Iterable
import gc
import torch
import whisper
from whisper.audio import load_audio
from pyannote.audio import Pipeline as PyannotePipeline
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sqlite3
from dotenv import load_dotenv
import os

from watcher.db import batched_insert, p_query

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
MODELS_FOLDER = Path("./models")


def prepare_transcription_model(
    model_name: str = "large-v3",
    device: str = DEVICE,
    download_root: Path = MODELS_FOLDER / "whisper",
) -> whisper.Whisper:
    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(
        model_name, device=device, download_root=str(download_root)
    )
    return model


def prepare_diarization_pipeline(
    device: str = DEVICE,
    auth_token: str = HF_TOKEN,
) -> PyannotePipeline:
    print("Loading PyAnnote speaker diarization model...")
    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=auth_token
    )
    pipeline = pipeline.to(torch.device(device))
    return pipeline


def prepare_embedding_model(
    device: str = DEVICE, model_folder: Path = MODELS_FOLDER
) -> SentenceTransformer:
    """Load embedding model into memory."""
    print("🔍 Loading SentenceTransformer embedder...")
    embedder = SentenceTransformer(
        str(model_folder / "bge-large-en-v1.5"), device=device
    )
    print("✅ Embedding model loaded successfully!")
    return embedder


def make_diarized_transcript(
    video_path: Path | str,
    whisper_model: whisper.Whisper,
    diarization_pipeline: PyannotePipeline = None,
) -> List[Dict[str, Any]]:
    """
    Drop-in replacement for your old function.
    Yields identical (chunk_tuple, embedding_tuple) format.
    """
    print("Extracting audio")
    video_path = str(video_path)
    audio: np.ndarray = load_audio(video_path)
    print(f"🎬 Transcribing video: {Path(video_path).name}")
    transcript_result: dict = whisper_model.transcribe(
        audio,
        language="en",
        word_timestamps=True,
        verbose=False,
        temperature=0.0,
        beam_size=5,
        best_of=5,
    )
    if diarization_pipeline:
        print("🗣️ Diarizing speakers")
        diarization: dict = diarization_pipeline(
            {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}
        )
        exclusive_diarization = diarization.exclusive_speaker_diarization
        transcript_result = transcript_result.get("segments", [])
        gc.collect()
        torch.cuda.empty_cache()
        transcript_result = pair_transcript_with_speaker(
            transcript_result, exclusive_diarization
        )
    return transcript_result


PairedResult = Tuple[str, float, float, str]


def pair_transcript_with_speaker(
    transcript_result: List[Dict[str, Any]],
    exclusive_diarization: List[Tuple[Any, str]],
    pause_threshold: float = 1.0,
) -> List[PairedResult]:
    """
    Marries words + diarization with crude lag fix.
    Inserts line breaks when the same speaker has a pause > pause_threshold.
    Prints beautifully in real time.
    """
    if not transcript_result:
        return []

    # Extract and sort all words
    words = sorted(
        (word for record in transcript_result for word in record.get("words", [])),
        key=lambda w: w["start"],
    )
    if not words:
        return []

    # Convert diarization (list or generator) to list once
    diarization = [(seg, speaker) for seg, speaker in exclusive_diarization]
    if not diarization:
        return []

    # Crude lag correction using first word and first diarization segment
    first_word_start = words[0]["start"]
    first_diar_start = diarization[0][0].start
    lag = first_word_start - first_diar_start

    print(
        f"Crude lag correction: {lag:+.3f}s (word @ {first_word_start:.2f}s → diar @ {first_diar_start:.2f}s)"
    )

    # Apply lag to all words
    for w in words:
        w["start"] += lag
        w["end"] += lag

    # Helper: emit a completed speaker block with nice formatting
    def emit(speaker: str, start: float, end: float, text: str):
        print(f"\n{speaker} ({start:.2f} → {end:.2f}):")
        # Split into paragraphs on double newlines, then print line-wrapped
        paragraphs = [p.strip() for p in text.strip().split("\n") if p.strip()]
        for para in paragraphs:
            words_in_para = para.split()
            for i in range(0, len(words_in_para), 16):
                print("    " + " ".join(words_in_para[i : i + 16]))
            print()  # extra blank line between paragraphs
        results.append((speaker, start, end, text.strip()))

    results: List[PairedResult] = []

    current_speaker = None
    current_start = None
    current_text = ""
    last_word_end = None  # track end time of previous word

    for word in words:
        w_start = word["start"]
        w_end = word["end"]
        w_text = (word.get("word") or word.get("text") or "").strip()
        if not w_text:
            continue

        # Find best speaker by max overlap
        best_speaker = "UNKNOWN"
        best_overlap = -1.0
        for seg, speaker in diarization:
            overlap_dur = max(0.0, min(w_end, seg.end) - max(w_start, seg.start))
            if overlap_dur > best_overlap:
                best_overlap = overlap_dur
                best_speaker = speaker

        # Speaker changed → emit previous block
        if current_speaker is not None and best_speaker != current_speaker:
            emit(current_speaker, current_start, last_word_end or w_start, current_text)
            current_text = ""
            last_word_end = None

        # New speaker starts
        if current_speaker != best_speaker:
            current_speaker = best_speaker
            current_start = w_start
            last_word_end = w_end
            current_text = w_text
            continue  # skip appending below — already added

        # Same speaker continues...

        # Check for significant pause (>1s) → insert paragraph break
        if last_word_end is not None and (w_start - last_word_end) >= pause_threshold:
            current_text += "\n\n"  # double newline = paragraph break

        # Append word (with space)
        current_text += " " + w_text
        last_word_end = w_end

    # Emit final speaker block
    if current_speaker and current_text.strip():
        final_end = last_word_end or words[-1]["end"]
        emit(current_speaker, current_start, final_end, current_text)

    return results


ChunkTableRecord = Tuple[int, str, float, float, str, str, Dict[str, Any]]


def chunk_diarized_transcript(
    transcript_result: PairedResult,
    video_id: int,
    metadata: Dict[str, Any] = None,
) -> Generator[ChunkTableRecord, None, None]:
    """
    yield consolidated transcript chunks for speakers
    """
    current_chunk_record = None
    for segment in tqdm(transcript_result, desc="Yielding chunks"):
        segment_speaker = segment[0]
        start_sec = segment[1]
        end_sec = segment[2]
        text = segment[3].strip()
        if current_chunk_record is None:
            current_chunk_record = [
                video_id,
                segment_speaker,
                start_sec,
                end_sec,
                text,
                "transcript",
                metadata,
            ]
            continue
        if segment_speaker != current_chunk_record[1]:
            finished_chunk_record = tuple(current_chunk_record)
            current_chunk_record = None
            yield finished_chunk_record
        else:
            current_chunk_record[3] = end_sec
            current_chunk_record[4] += " " + text
            continue


def save_chunks_to_db(
    conn: sqlite3.Connection,
    chunks: Iterable[ChunkTableRecord],
    batch_size: int = 100,
) -> None:
    """Save chunks to the database in batches."""
    batched_insert(
        conn,
        "chunks",
        "insert_chunk",
        chunks,
        batch_size=batch_size,
    )


ChunkEmbedding = Tuple[int, bytes]


def chunk_embeddings(
    chunks_with_ids: Iterable[Tuple[int, str]], embedder: SentenceTransformer
) -> Generator[ChunkEmbedding, None, None]:
    for chunk in chunks_with_ids:
        text = chunk[1].strip()
        chunk_id = chunk[0]
        if not text:
            continue
        emb = embedder.encode(text, normalize_embeddings=True)
        emb_bytes = np.array(emb, dtype=np.float32).tobytes()
        embedding_tuple = (
            chunk_id,
            emb_bytes,
        )
        yield embedding_tuple
    gc.collect()
    torch.cuda.empty_cache()


def transcribe_videos_to_db(
    conn: sqlite3.Connection,
    whisper_model: whisper.Whisper = None,
    diarization_pipeline: PyannotePipeline = None,
) -> None:
    videos_to_transcribe = p_query(conn, "videos", "get_videos_to_transcribe", ())
    video_count = len(videos_to_transcribe)
    if video_count == 0:
        print("No videos to transcribe.")
        return
    print(f"Preparing to Transcribe {video_count} videos...")
    whisper_model = (
        prepare_transcription_model() if not whisper_model else whisper_model
    )
    diarization_pipeline = (
        prepare_diarization_pipeline()
        if not diarization_pipeline
        else diarization_pipeline
    )
    for video_row in videos_to_transcribe:
        video_id = video_row["video_id"]
        video_path = video_row["video_file_path"]
        print(f"Transcribing video ID {video_id} at path {video_path}...")
        transcript_result = make_diarized_transcript(
            video_path,
            whisper_model,
            diarization_pipeline=diarization_pipeline,
        )
        chunk_generator = chunk_diarized_transcript(
            transcript_result,
            video_id,
        )
        save_chunks_to_db(conn, chunk_generator, batch_size=100)


def embed_chunks_in_db(
    conn: sqlite3.Connection,
    embedder: SentenceTransformer = None,
    batch_size: int = 256,
) -> None:
    """Embed chunks and save embeddings to the database in batches."""
    count_of_chunks_without_embeddings = p_query(
        conn, "chunk_vectors", "get_chunks_without_embeddings_count"
    )[0]["count"]
    print(
        f"Embedding {count_of_chunks_without_embeddings} chunks without embeddings..."
    )
    if count_of_chunks_without_embeddings == 0:
        print("No chunks to embed.")
        return
    if not embedder:
        embedder = prepare_embedding_model()
    for _ in tqdm(
        range(0, count_of_chunks_without_embeddings, batch_size),
        desc="Embedding chunks",
    ):
        chunks_to_embed = p_query(
            conn,
            "chunk_vectors",
            "get_chunks_without_embeddings",
            (batch_size,),
        )
        chunk_ids_texts = [(row["chunk_id"], row["text"]) for row in chunks_to_embed]
        embedding_tuples = chunk_embeddings(chunk_ids_texts, embedder)
        batched_insert(
            conn,
            "chunk_vectors",
            "insert_embedding",
            embedding_tuples,
            batch_size=batch_size,
        )


__all__ = ["transcribe_videos_to_db", "embed_chunks_in_db"]

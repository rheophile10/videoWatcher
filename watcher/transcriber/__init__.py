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
from watcher.db.chunks import export_today_chunk_hits
from tqdm import tqdm
import sqlite3
from dotenv import load_dotenv
import os
from pydub import AudioSegment

from watcher.llm.resources import register_loaded_model
from watcher.db import batched_insert, p_query, dicts_to_csv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
MODELS_FOLDER = Path("./models")


def prepare_transcription_model(
    model_name: str = "large-v2",
    device: str = DEVICE,
    download_root: Path = MODELS_FOLDER / "whisper",
) -> whisper.Whisper:
    register_loaded_model(f"whisper:{model_name}", estimated_gb=2.5)
    model = whisper.load_model(
        model_name, device=device, download_root=str(download_root)
    )
    return model


def prepare_diarization_pipeline(
    device: str = DEVICE,
    auth_token: str = HF_TOKEN,
) -> PyannotePipeline:
    register_loaded_model("pyannote/speaker-diarization-3.1", estimated_gb=1.1)
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


def transcribe_with_chunks(
    model: whisper.Whisper,
    video_path: str,
    chunk_length_ms: int = 30_000,  # 30 seconds
    overlap_ms: int = 2_000,  # 2 seconds overlap
    language: str = "en",
    temperature: float = 0.2,
    beam_size: int = 1,
    best_of: int = 5,
    patience: float = 2.0,
    compression_ratio_threshold: float = 2.4,
    no_speech_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Transcribe long audio/video reliably using fixed-size chunks + overlap.
    Returns a list of segments with accurate global timestamps.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"File not found: {video_path}")

    print(
        f"Loading audio from: {video_path.name} ({os.path.getsize(video_path)/1e6:.1f} MB)"
    )
    audio = AudioSegment.from_file(video_path)
    total_duration_ms = len(audio)
    total_seconds = total_duration_ms / 1000

    print(
        f"Audio duration: {total_seconds:.1f} seconds → "
        f"~{total_duration_ms // chunk_length_ms + 1} chunks"
    )

    segments = []
    temp_files = []

    # Pre-calculate all chunk boundaries
    chunk_starts = list(range(0, total_duration_ms, chunk_length_ms - overlap_ms))
    if chunk_starts[-1] + chunk_length_ms < total_duration_ms:
        chunk_starts.append(total_duration_ms - chunk_length_ms)  # final chunk

    # Main progress bar
    with tqdm(
        total=len(chunk_starts), desc="Transcribing chunks", unit="chunk"
    ) as pbar:
        for i, chunk_start_ms in enumerate(chunk_starts):
            chunk_end_ms = min(chunk_start_ms + chunk_length_ms, total_duration_ms)

            # Adjust start with overlap (except first chunk)
            if i > 0:
                extract_start = chunk_start_ms - overlap_ms
            else:
                extract_start = chunk_start_ms

            chunk = audio[extract_start:chunk_end_ms]
            temp_path = Path(f"temp_chunk_{os.getpid()}_{i}.wav")  # unique per process
            chunk.export(temp_path, format="wav")
            temp_files.append(temp_path)

            # Transcribe
            result = model.transcribe(
                str(temp_path),
                language=language,
                word_timestamps=False,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                compression_ratio_threshold=compression_ratio_threshold,
                no_speech_threshold=no_speech_threshold,
            )

            # Adjust timestamps to global time
            offset_seconds = extract_start / 1000.0
            for seg in result.get("segments", []):
                seg["start"] += offset_seconds
                seg["end"] += offset_seconds
                segments.append(
                    {
                        "start": round(seg["start"], 3),
                        "end": round(seg["end"], 3),
                        "text": seg["text"].strip(),
                    }
                )

            pbar.update(1)
            pbar.set_postfix(
                {
                    "time": f"{chunk_end_ms/1000:.1f}s",
                    "chunks_left": len(chunk_starts) - (i + 1),
                }
            )

    # === Optional: Deduplicate overlapping text (very light & effective) ===
    if len(segments) > 1 and overlap_ms > 0:
        cleaned = []
        last_text = ""
        for seg in segments:
            # If this segment starts near the end of previous one and text overlaps heavily
            if (
                cleaned
                and seg["start"] < cleaned[-1]["end"] + 0.5
                and seg["text"] in cleaned[-1]["text"] + " "
            ):  # rough but works great
                # Merge or skip duplicate
                cleaned[-1]["end"] = max(cleaned[-1]["end"], seg["end"])
                cleaned[-1]["text"] = (
                    cleaned[-1]["text"]
                    + " "
                    + seg["text"].split(last_text, 1)[-1].strip()
                )
            else:
                cleaned.append(seg)
            last_text = seg["text"]
        segments = cleaned

    # === Cleanup ===
    for temp_path in temp_files:
        try:
            temp_path.unlink()
        except:
            pass

    print(f"Transcription complete! {len(segments)} clean segments.")
    return segments


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
    transcript_result = transcribe_with_chunks(whisper_model, video_path=video_path)
    if diarization_pipeline:
        print("🗣️ Diarizing speakers")
        diarization: dict = diarization_pipeline(
            {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}
        )
        exclusive_diarization = diarization.exclusive_speaker_diarization
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
) -> List[PairedResult]:
    """
    Marries words + diarization with crude lag fix.
    Inserts line breaks when the same speaker has a pause > pause_threshold.
    Prints beautifully in real time.
    """
    if not transcript_result:
        return []

    # Extract and sort all words
    # words = sorted(
    #     (word for record in transcript_result for word in record.get("words", [])),
    #     key=lambda w: w["start"],
    # )
    # if not words:
    #     return []

    # Convert diarization (list or generator) to list once
    diarization = [(seg, speaker) for seg, speaker in exclusive_diarization]
    dicts_to_csv(
        [
            {"start": seg.start, "end": seg.end, "speaker": spk}
            for seg, spk in diarization
        ],
        f"diarization_segments",
    )
    # dicts_to_csv(
    #     [
    #         {"start": w["start"], "end": w["end"], "word": w.get("word", "")}
    #         for w in words
    #     ],
    #     f"transcript_words",
    # )
    dicts_to_csv(
        transcript_result,
        f"transcript_text",
    )
    if not diarization:
        return []

    results: List[PairedResult] = []
    for text in transcript_result:
        text_start = text["start"]
        text_end = text["end"]
        text_str = text["text"].strip()
        # shitty speaker identification aglo
        speaker_label = "Unknown"
        for seg, speaker in diarization:
            if seg.start <= text_start <= seg.end or seg.start <= text_end <= seg.end:
                speaker_label = speaker
                break
        results.append((speaker_label, text_start, text_end, text_str))

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
    export_chunks_after_each_video: bool = False,
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
        if export_chunks_after_each_video:
            export_today_chunk_hits(conn, video_id=video_id)


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

import csv
import os
from watcher.llm import generate_memos


def generate_and_export_memos(chunks, output_csv_path, platform="X"):
    """
    Generate memos from a list of chunks and export them to a CSV file.
    """
    # Generate memos
    memos = generate_memos(chunks, platform=platform, model="openai:gpt-4o")

    # Export to CSV
    if memos:
        fieldnames = memos[0].keys()
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(memos)
        print(f"Memos exported to {output_csv_path}")
    else:
        print("No memos to export.")
    return memos


# Path to the CSV file
input_csv_file_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "exports",
    "export_today_chunk_hits20251124_150312.csv",
)

# Read the CSV and treat as list of dicts (mimicking sqlite3 rows)
input_chunks = []
try:
    with open(input_csv_file_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            input_chunks.append(row)
except FileNotFoundError:
    print(f"File not found: {input_csv_file_path}")
    input_chunks = []  # Empty list if file doesn't exist

# Test the function
output_path = os.path.join(
    os.path.dirname(__file__), "..", "exports", "generated_memos.csv"
)
generated_memos = generate_and_export_memos(input_chunks, output_path)

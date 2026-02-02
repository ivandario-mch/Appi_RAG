import os
import argparse
import sys

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion import IngestionPipeline
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline.")
    parser.add_argument("--data_dir", default="data", help="Directory containing PDF files to ingest.")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return

    # Validate config
    try:
        Config.validate()
    except Exception as e:
        print(f"Configuration Error: {e}")
        return

    # Initialize Pipeline
    try:
        pipeline = IngestionPipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    # Process files
    files = [f for f in os.listdir(data_dir) if f.casefold().endswith(".pdf")]
    
    if not files:
        print(f"No PDF files found in '{data_dir}'.")
        return

    process_files(files, data_dir, pipeline)

def process_files(files, data_dir, pipeline):
    print(f"Found {len(files)} PDF file(s) in '{data_dir}'. Starting ingestion...")
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        try:
            pipeline.process_pdf(file_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\nBatch Ingestion Finished.")

if __name__ == "__main__":
    main()

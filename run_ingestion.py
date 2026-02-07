import os
import argparse
import sys

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion import IngestionPipeline
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline with Qdrant.")
    parser.add_argument("--data_dir", default="data", help="Directory containing PDF files to ingest.")
    parser.add_argument("--reset", action="store_true", help="Delete existing collection before ingestion.")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics only (no ingestion).")
    args = parser.parse_args()

    # Validate config
    try:
        Config.validate()
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        print("\nMake sure your .env file has:")
        print("  - QDRANT_URL")
        print("  - QDRANT_API_KEY")
        print("  - GROQ_API_KEY")
        return

    # Initialize Pipeline
    try:
        pipeline = IngestionPipeline()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return

    # Show stats only
    if args.stats:
        stats = pipeline.get_collection_stats()
        print("\n📊 Collection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

    # Reset collection if requested
    if args.reset:
        print("\n⚠️  Resetting collection...")
        pipeline.delete_collection()
        pipeline._create_collection_if_not_exists()

    # Check data directory
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory '{data_dir}' does not exist.")
        print(f"Creating directory: {data_dir}")
        os.makedirs(data_dir)
        print(f"Please add PDF files to '{data_dir}' and run again.")
        return

    # Process files
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    
    if not files:
        print(f"⚠️  No PDF files found in '{data_dir}'.")
        print(f"Please add PDF files to '{data_dir}' and run again.")
        return

    process_files(files, data_dir, pipeline)
    
    # Show final stats
    print("\n" + "="*60)
    stats = pipeline.get_collection_stats()
    print("📊 Final Collection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*60)

def process_files(files, data_dir, pipeline):
    print(f"\n📁 Found {len(files)} PDF file(s) in '{data_dir}'")
    print("="*60)
    print("Starting batch ingestion...\n")
    
    successful = 0
    failed = 0
    
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(data_dir, filename)
        print(f"[{i}/{len(files)}] Processing: {filename}")
        print("-"*60)
        
        try:
            pipeline.process_pdf(file_path)
            successful += 1
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            failed += 1
        
        print()

    print("="*60)
    print(f"✅ Batch Ingestion Finished")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print("="*60)

if __name__ == "__main__":
    main()
"""
extract_topics_from_files.py

Extracts topics from each SOW file and creates a JSON mapping:
{
  "0000": ["Scope of Work", "Deliverables", "Payment Terms"],
  "0001": ["Technical Requirements", "Intellectual Property"],
  ...
}
"""

import sys
import os
import asyncio
import json
import glob
from typing import Dict, List
from tqdm.asyncio import tqdm
from code.helpers.llm import SimpleLLMCaller




async def extract_topics_from_file(llm_caller: SimpleLLMCaller, 
                                   file_path: str, 
                                   model: str) -> List[str]:
    """
    Extract topics from a single SOW file.
    
    Args:
        llm_caller: The LLM caller instance
        file_path: Path to the SOW file
        model: Model name to use
        
    Returns:
        List of topic names found in the file
    """
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    # Create prompt for topic extraction
    prompt = f"""You are analyzing a Statement of Work (SOW) document to identify its organizational structure.

Your task: Identify the types of SECTIONS or CATEGORIES of information present in this document, not the specific technical details.
Each topic is supposed to be at max 2-3 words long.

Think abstractly about the document's structure:
- Don't describe WHAT is being built → describe what TYPE of information that section contains
- Don't list the technologies → identify what PURPOSE that content serves in the contract
- Focus on the ROLE each part plays in the overall agreement

For example:
- A section listing items to be provided → that's a category about "what will be provided"
- A section with dates and amounts for billing → that's a category about "financial arrangements"  
- A section describing the work activities → that's a category about "work to be done"

Use this pattern: ask yourself "what is the PURPOSE of this information in a business agreement?" rather than "what is the specific content?"

Document:
---
{content}
---

Return ONLY a comma-separated list of 5-7 abstract category names that describe the TYPES of sections in this document.
Use terminology that describes the structural purpose, not the specific content:"""

    generation_params = {
        "temperature": 0.3,
        "client_identifier": "backend-core-devs",
        "provider": "OPEN_AI",
        "max_tokens": 500
    }
    
    try:
        response = await llm_caller.call_and_get_text(
            prompt=prompt,
            model=model,
            generation_params=generation_params
        )
        
        if response and response.strip():
            # Parse comma-separated topics
            topics = [t.strip() for t in response.split(',') if t.strip()]
            # Remove any numbering or bullets if present
            topics = [t.lstrip('0123456789.-) ') for t in topics]
            return topics
        else:
            return []
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


async def process_all_files(input_dir: str, 
                           output_file: str,
                           model: str = "gpt-4.1-mini",
                           max_concurrency: int = 10,
                           limit: int = None):
    """
    Process all SOW files and extract topics.
    
    Args:
        input_dir: Directory containing SOW files
        output_file: Output JSON file path
        model: Model name to use
        max_concurrency: Maximum concurrent LLM calls
        limit: Optional limit on number of files to process (for testing)
    """
    # Initialize LLM caller
    llm_caller = SimpleLLMCaller(max_retries=3, semaphore=max_concurrency)
    
    # Find all SOW files
    pattern = os.path.join(input_dir, "sow_*.txt")
    all_files = sorted(glob.glob(pattern))
    
    if limit:
        all_files = all_files[:limit]
    
    print(f"Found {len(all_files)} SOW files to process")
    print(f"Using model: {model}")
    print(f"Max concurrency: {max_concurrency}")
    print("=" * 80)
    
    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(max_concurrency)
    
    async def process_file(file_path: str) -> tuple:
        """Process a single file with semaphore."""
        async with sem:
            # Extract file ID (e.g., "0000" from "sow_0000.txt")
            basename = os.path.basename(file_path)
            file_id = basename.replace("sow_", "").replace(".txt", "")
            
            topics = await extract_topics_from_file(llm_caller, file_path, model)
            return (file_id, topics)
    
    # Process all files with progress bar
    results = {}
    tasks = [process_file(f) for f in all_files]
    
    print("\nProcessing files...")
    for coro in tqdm.as_completed(tasks, total=len(tasks)):
        file_id, topics = await coro
        results[file_id] = topics
    
    # Sort results by file ID
    results = dict(sorted(results.items()))
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Topics extracted from {len(results)} files")
    print(f"✓ Results saved to: {output_file}")
    
    # Print some statistics
    total_topics = sum(len(topics) for topics in results.values())
    avg_topics = total_topics / len(results) if results else 0
    print(f"\n📊 Statistics:")
    print(f"   - Total files processed: {len(results)}")
    print(f"   - Average topics per file: {avg_topics:.1f}")
    print(f"   - Total topic instances: {total_topics}")
    
    # Show a sample
    print(f"\n📋 Sample results (first 5 files):")
    for file_id, topics in list(results.items())[:5]:
        print(f"   {file_id}: {topics}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract topics from SOW files")
    parser.add_argument(
        "--input-dir",
        default="./generated_data",
        help="Directory containing SOW files (default: ./generated_data)"
    )
    parser.add_argument(
        "--output",
        default="./generated_data/sow_topics.json",
        help="Output JSON file (default: ./generated_data/sow_topics.json)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model to use (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent LLM calls (default: 10)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(process_all_files(
        input_dir=args.input_dir,
        output_file=args.output,
        model=args.model,
        max_concurrency=args.concurrency,
        limit=args.limit
    ))


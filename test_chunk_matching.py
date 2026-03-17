#!/usr/bin/env python3
"""
Test script to validate the new sentence-to-chunk matching functionality.
This demonstrates how the new implementation selects only relevant chunks
instead of all chunks from a page.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vectorizerE import find_supporting_chunks_for_text, format_chunk_references

def test_chunk_matching():
    """Test the chunk matching functionality with sample data."""
    
    # Sample summary text
    summary_text = """The document provides comprehensive information about cancer care policy terms and coverage options. It details the premium payment structure and outlines specific exclusions for pre-existing conditions."""
    
    # Sample chunk data (simulating real chunk content)
    sample_chunks = [
        {
            "chunk_index": 0,
            "content": "HDFC Life Cancer Care Policy provides comprehensive coverage for cancer treatment expenses including hospitalization and medical procedures.",
        },
        {
            "chunk_index": 1, 
            "content": "The premium payment structure includes annual, semi-annual, quarterly and monthly payment options with different benefits for each.",
        },
        {
            "chunk_index": 2,
            "content": "Policy exclusions include pre-existing medical conditions that were diagnosed before the policy start date.",
        },
        {
            "chunk_index": 3,
            "content": "The nominee details section describes the process for appointing beneficiaries and their rights under the policy.",
        },
        {
            "chunk_index": 4,
            "content": "General terms and conditions include policy governance, legal jurisdiction and dispute resolution procedures.",
        },
        {
            "chunk_index": 5,
            "content": "Investment options are not applicable as this is a non-participating insurance product without investment features.",
        }
    ]
    
    print("=== Chunk Matching Test ===")
    print(f"Summary: {summary_text}")
    print("\nAvailable chunks:")
    for i, chunk in enumerate(sample_chunks):
        print(f"  C{chunk['chunk_index']}: {chunk['content'][:80]}...")
    
    # Test the intelligent matching
    relevant_chunks = find_supporting_chunks_for_text(
        summary_text, 
        sample_chunks, 
        top_k_per_sentence=3,
        min_overlap_score=0.1
    )
    
    print(f"\nIntelligent matching selected chunks: {relevant_chunks}")
    
    # Expected: Should select chunks 0 (cancer care), 1 (premium payment), 2 (exclusions)
    # Should NOT select chunks 3 (nominee), 4 (general terms), 5 (investment)
    
    chunk_ref = format_chunk_references(relevant_chunks)
    print(f"Formatted reference: {chunk_ref}")
    
    print(f"\nSummary with references: {summary_text}{chunk_ref}")
    
    # Verify results
    expected_chunks = {0, 1, 2}  # Chunks that should match the summary content
    selected_chunks = set(relevant_chunks)
    
    correct_selections = selected_chunks & expected_chunks
    incorrect_selections = selected_chunks - expected_chunks
    
    print(f"\n=== Results ===")
    print(f"Correctly selected: {list(correct_selections)}")
    print(f"Incorrectly selected: {list(incorrect_selections)}")
    print(f"Precision: {len(correct_selections)}/{len(selected_chunks)} = {len(correct_selections)/max(1,len(selected_chunks)):.2%}")
    
    # Test passes if we get good precision (>= 70%) and recall (>= 2 correct chunks)
    precision = len(correct_selections) / max(1, len(selected_chunks))
    return precision >= 0.7 and len(correct_selections) >= 2

if __name__ == "__main__":
    success = test_chunk_matching()
    if success:
        print("\n[SUCCESS] Test PASSED - Chunk matching is working correctly!")
    else:
        print("\n[FAILED] Test FAILED - Check chunk matching logic")
    sys.exit(0 if success else 1)
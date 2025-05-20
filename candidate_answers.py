from pathlib import Path
import random
from typing import Dict, List, Set
import pandas as pd
from file_manager import FileManager
from scat_utils import get_deterministic_instances
import argparse

class CandidateAnswers:
    def __init__(self, file_manager: FileManager):
        self.fm = file_manager

    def get_all_candidates(self, letter: str, category: str, count_threshold: int = 0) -> Set[str]:
        """
        Get all unique candidate answers for a given letter and category.
        Combines answers from all models and verified sources.
        
        Args:
            letter: The starting letter
            category: The category name
            count_threshold: Minimum count threshold (summed across all models) to include an answer
        """
        # Dictionary to store summed counts for each candidate
        candidate_counts = {}

        # Get all samples
        samples_df = self.fm.get_all_samples(letter=letter, category=category)
        for _, row in samples_df.iterrows():
            try:
                samples = self.fm.load_samples(
                    letter=row['letter'],
                    category=row['category'],
                    model_name=row['model'],
                    temp=row['temperature']
                )
                if 'dist' in samples:
                    for answer, count in samples['dist'].items():
                        candidate_counts[answer] = candidate_counts.get(answer, 0) + count
            except Exception as e:
                print(f"Error loading samples for {row['fname']}: {e}")

        # Get all verified answers
        verified_df = self.fm.get_all_verified(letter=letter, category=category)
        verified_answers = set()
        for _, row in verified_df.iterrows():
            try:
                verified = self.fm.load_verified(
                    letter=row['letter'],
                    category=row['category'],
                    v_name=row['verifier']
                )
                if 'yes' in verified:
                    verified_answers.update(verified['yes'])
            except Exception as e:
                print(f"Error loading verified answers for {row['fname']}: {e}")
        
        # Filter answers to only include verified ones
        candidate_counts = {k: v for k, v in candidate_counts.items() if k in verified_answers}

        # Filter candidates based on count threshold
        candidates = {answer for answer, count in candidate_counts.items() 
                     if count >= count_threshold}
        print(min(candidate_counts.values()), max(candidate_counts.values()))

        # Save the answer set
        self.fm.write_answer_set(letter, category, count_threshold, candidates)

        return candidates

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Generate candidate answers for scattergories')
    parser.add_argument('--count-threshold', type=int, default=1,
                      help='Minimum count threshold for including an answer (default: 1)')
    args = parser.parse_args()

    # Initialize FileManager
    fm = FileManager.from_base(Path('./'))
    
    # Get deterministic instances
    instances = get_deterministic_instances(1)
    
    # Load candidate answers for each instance
    for letter, category in instances:
        candidates = ca.get_all_candidates(letter, category, count_threshold=args.count_threshold)
        print(f"\n{letter} {category}:")
        print(f"Found {len(candidates)} candidate answers")

if __name__ == '__main__':
    # Initialize file manager and candidate answers
    fm = FileManager.from_base(Path('./'))
    ca = CandidateAnswers(fm)
    
    # Process each instance
    main()
        
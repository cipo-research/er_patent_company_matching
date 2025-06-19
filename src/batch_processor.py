# src/batch_processor.py
"""
Batch processing utilities for large-scale matching operations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

class BatchProcessor:
    """Class to handle large-scale batch processing of patent-stock matching"""
    
    def __init__(self, batch_size=1000, n_workers=None):
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
    
    def create_batches(self, df, batch_size=None):
        """
        Split dataframe into batches for processing
        
        Args:
            df (pd.DataFrame): DataFrame to split
            batch_size (int): Size of each batch
            
        Returns:
            list: List of DataFrame batches
        """
        batch_size = batch_size or self.batch_size
        batches = []
        
        for i in range(0, len(df), batch_size):
            batch = df[i:i + batch_size].copy()
            batches.append(batch)
        
        self.logger.info(f"Created {len(batches)} batches of size {batch_size}")
        return batches
    
    def process_batch(self, patent_batch, stock_df, cleaner, matcher):
        """
        Process a single batch of patent data
        
        Args:
            patent_batch (pd.DataFrame): Batch of patent data
            stock_df (pd.DataFrame): Complete stock data
            cleaner: Name cleaner instance
            matcher: Matcher instance
            
        Returns:
            pd.DataFrame: Matches for this batch
        """
        try:
            # Clean the batch
            patent_batch_cleaned = cleaner.clean_patent_names(patent_batch)
            
            # Find matches for this batch
            batch_matches = matcher.find_matches(patent_batch_cleaned, stock_df)
            
            return batch_matches
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def parallel_process(self, patent_df, stock_df, cleaner, matcher, save_intermediate=True, output_dir=None):
        """
        Process patent data in parallel batches
        
        Args:
            patent_df (pd.DataFrame): Patent data
            stock_df (pd.DataFrame): Stock data
            cleaner: Name cleaner instance
            matcher: Matcher instance
            save_intermediate (bool): Whether to save intermediate results
            output_dir (str): Directory for intermediate files
            
        Returns:
            pd.DataFrame: Combined results
        """
        # Clean stock data once (it's the same for all batches)
        self.logger.info("Cleaning stock data...")
        stock_df_cleaned = cleaner.clean_stock_names(stock_df)
        
        # Create batches
        patent_batches = self.create_batches(patent_df)
        all_results = []
        
        self.logger.info(f"Starting parallel processing with {self.n_workers} workers...")
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self.process_batch, batch, stock_df_cleaned, cleaner, matcher): i
                for i, batch in enumerate(patent_batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_result = future.result()
                    all_results.append(batch_result)
                    
                    self.logger.info(f"Completed batch {batch_idx + 1}/{len(patent_batches)} - Found {len(batch_result)} matches")
                    
                    # Save intermediate results if requested
                    if save_intermediate and output_dir:
                        intermediate_file = Path(output_dir) / f"batch_{batch_idx:04d}_matches.csv"
                        batch_result.to_csv(intermediate_file, index=False)
                    
                except Exception as e:
                    self.logger.error(f"Batch {batch_idx} failed with error: {str(e)}")
        
        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Sort combined results
            combined_results = combined_results.sort_values(
                ['match_score', 'confidence_mean', 'total_patents'], 
                ascending=[False, False, False]
            ).reset_index(drop=True)
            
            self.logger.info(f"Parallel processing completed. Total matches: {len(combined_results)}")
            return combined_results
        else:
            self.logger.warning("No results from parallel processing")
            return pd.DataFrame()


# Example usage script
def run_example():
    """
    Example script showing how to use the matching system
    """
    from name_cleaner import NameCleaner
    from matcher import CompanyMatcher
    from match_analyzer import MatchAnalyzer
    from data_loader import DataLoader
    from utils import setup_logging
    
    # Setup
    logger = setup_logging()
    logger.info("Running patent-stock matching example")
    
    # Load sample data (you would replace this with your actual data loading)
    sample_patent_data = pd.DataFrame({
        'company': [
            'International Business Machines Corp',
            'Microsoft Corporation',
            'Google Inc',
            'Apple Inc',
            'Amazon.com Inc',
            'Tesla Motors Inc',
            'Johnson & Johnson',
            'Pfizer Inc'
        ],
        'total_patents': [27000, 15000, 12000, 8000, 5000, 3000, 18000, 12000]
    })
    
    sample_stock_data = pd.DataFrame({
        'ticker': ['IBM', 'MSFT', 'GOOGL', 'AAPL', 'AMZN', 'TSLA', 'JNJ', 'PFE'],
        'company_name': [
            'International Business Machines',
            'Microsoft Corp',
            'Alphabet Inc',
            'Apple Inc',
            'Amazon.com Inc',
            'Tesla Inc',
            'Johnson Johnson',
            'Pfizer Inc'
        ],
        'exchange': ['NYSE'] * 8,
        'category': ['Technology', 'Technology', 'Technology', 'Technology', 
                    'Consumer Discretionary', 'Consumer Discretionary', 
                    'Healthcare', 'Healthcare'],
        'country': ['US'] * 8
    })
    
    # Processing
    cleaner = NameCleaner()
    stock_data_cleaned = cleaner.clean_stock_names(sample_stock_data)
    patent_data_cleaned = cleaner.clean_patent_names(sample_patent_data)
    
    # Matching
    matcher = CompanyMatcher(
        exact_threshold=1.0,
        fuzzy_threshold=0.75,
        sector_boost=0.1
    )
    
    matches = matcher.find_matches(patent_data_cleaned, stock_data_cleaned)
    
    # Analysis
    analyzer = MatchAnalyzer()
    report = analyzer.generate_match_report(matches)
    print(report)
    
    # Display results
    print("\n=== MATCHING RESULTS ===")
    display_cols = [
        'applicant_name_patstat', 'company_name_stock', 'ticker_stock',
        'match_score', 'sector_match', 'confidence_mean', 'total_patents'
    ]
    print(matches[display_cols].to_string(index=False, float_format='%.3f'))
    
    return matches

if __name__ == "__main__":
    # Run the example
    matches = run_example()


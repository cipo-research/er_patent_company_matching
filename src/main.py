# src/main.py
"""
main script with all improvements
"""

import os
import sys
import pandas as pd
from pathlib import Path
import time

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from name_cleaner import NameCleaner
from matcher import CompanyMatcher
from utils import setup_logging, create_output_directory

def main():
    """main execution function"""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting patent-stock company matching process")
    
    # Record start time
    start_time = time.time()
    
    # Sample settings
    SAMPLE_MODE = True
    PATENT_SAMPLE_SIZE = 1000
    STOCK_SAMPLE_SIZE = 100000
    
    # Define file paths
    BASE_DIR = Path(__file__).parent.parent
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    
    create_output_directory(PROCESSED_DATA_DIR)
    
    stock_file = RAW_DATA_DIR / "Yahoo Ticker Symbols - September 2017.xlsx"
    patent_file = PROCESSED_DATA_DIR / "table1.csv"
    
    if SAMPLE_MODE:
        output_file = PROCESSED_DATA_DIR / f"patent_stock_matches_sample_{PATENT_SAMPLE_SIZE}.csv"
        detailed_output_file = PROCESSED_DATA_DIR / f"detailed_matches_sample_{PATENT_SAMPLE_SIZE}.csv"
    else:
        output_file = PROCESSED_DATA_DIR / "patent_stock_matches_full.csv"
        detailed_output_file = PROCESSED_DATA_DIR / "detailed_matches_full.csv"
    
    try:
        # Load data
        logger.info("Loading data...")
        load_start = time.time()
        data_loader = DataLoader()
        stock_data = data_loader.load_stock_data(stock_file)
        patent_data = data_loader.load_patent_data(patent_file)
        logger.info(f"Data loading completed in {time.time() - load_start:.2f} seconds")
        
        # Apply sampling if in sample mode
        if SAMPLE_MODE:
            patent_data = patent_data.head(PATENT_SAMPLE_SIZE)
            stock_data = stock_data.head(STOCK_SAMPLE_SIZE)
            logger.info(f"Sampled data: {len(stock_data)} stock entries, {len(patent_data)} patent applicants")
        
        # Cleaning
        logger.info("Cleaning company names...")
        clean_start = time.time()
        cleaner = NameCleaner()
        stock_data = cleaner.clean_stock_names(stock_data)
        patent_data = cleaner.clean_patent_names(patent_data)
        logger.info(f"Cleaning completed in {time.time() - clean_start:.2f} seconds")
        
        # Matching
        logger.info("Matching with sector awareness...")
        match_start = time.time()
        matcher = CompanyMatcher(
            exact_threshold=1.0,
            fuzzy_threshold=0.85,
            sector_boost=0.1
        )
        matches = matcher.find_matches(patent_data, stock_data)
        logger.info(f"Matching completed in {time.time() - match_start:.2f} seconds")
        
        # Save detailed results
        logger.info(f"Found {len(matches)} matches. Saving detailed results...")
        matches.to_csv(detailed_output_file, index=False)
        
        # Create summary version
        summary_columns = [
            'applicant_name_patstat', 'company_name_stock', 'ticker_stock',
            'match_score', 'match_type', 'sector_match', 'parent_match',
            'confidence_mean', 'confidence_lower', 'confidence_upper',
            'total_patents'
        ]
        summary_matches = matches[summary_columns].copy()
        summary_matches.to_csv(output_file, index=False)
        
        # Summary statistics
        print(f"\n=== MATCHING SUMMARY ===")
        if SAMPLE_MODE:
            print(f"SAMPLE MODE - Top {PATENT_SAMPLE_SIZE} patent applicants")
        print(f"Total patent applicants: {len(patent_data)}")
        print(f"Total stock companies: {len(stock_data)}")
        print(f"Successful matches: {len(matches)}")
        print(f"Match rate: {len(matches)/len(patent_data)*100:.1f}%")
        
        # Match type breakdown
        exact_matches = len(matches[matches['match_type'] == 'exact'])
        fuzzy_matches = len(matches[matches['match_type'] == 'fuzzy'])
        print(f"Exact matches: {exact_matches}")
        print(f"Fuzzy matches: {fuzzy_matches}")
        
        # Sector matching statistics
        sector_matches = len(matches[matches['sector_match'] == True])
        parent_matches = len(matches[matches['parent_match'] == True])
        print(f"Same-sector matches: {sector_matches} ({sector_matches/len(matches)*100:.1f}%)")
        print(f"Parent-subsidiary matches: {parent_matches} ({parent_matches/len(matches)*100:.1f}%)")
        
        # Confidence statistics
        high_confidence = len(matches[matches['confidence_lower'] >= 0.85])
        medium_confidence = len(matches[(matches['confidence_lower'] >= 0.70) & (matches['confidence_lower'] < 0.85)])
        print(f"High confidence matches (CI lower >= 0.85): {high_confidence}")
        print(f"Medium confidence matches (CI lower 0.70-0.85): {medium_confidence}")
        
        # Performance statistics
        total_time = time.time() - start_time
        print(f"\n=== PERFORMANCE ===")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Matches per second: {len(matches)/total_time:.2f}")
        print(f"Comparisons per second: {(len(patent_data) * len(stock_data))/total_time:.2f}")
        
        print(f"\nDetailed results: {detailed_output_file}")
        print(f"Summary results: {output_file}")
        
        # Show top matches
        if len(matches) > 0:
            print(f"\n=== TOP 10 MATCHES ===")
            display_columns = [
                'applicant_name_patstat', 'company_name_stock', 'ticker_stock', 
                'match_score', 'match_type', 'sector_match', 'parent_match',
                'confidence_mean', 'total_patents'
            ]
            top_matches = matches[display_columns].head(10)
            print(top_matches.to_string(index=False, float_format='%.3f'))
            
            # Sector distribution
            print(f"\n=== SECTOR DISTRIBUTION ===")
            sector_dist = matches['patent_sector'].value_counts()
            print(sector_dist.head(10))
        
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
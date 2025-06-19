# src/data_loader.py
"""
Data loading utilities for patent and stock data
"""

import pandas as pd
import logging
from pathlib import Path

class DataLoader:
    """Class to handle loading of patent and stock data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_stock_data(self, file_path):
        """
        Load stock ticker data from Excel file
        
        Args:
            file_path (Path): Path to the Excel file
            
        Returns:
            pd.DataFrame: Cleaned stock data
        """
        try:
            # Based on your header analysis, the actual data starts at row 3 (0-indexed)
            df = pd.read_excel(file_path, header=3)
            
            # Select only the columns we need
            stock_cols = ['Ticker', 'Name', 'Exchange', 'Category Name', 'Country']
            df = df[stock_cols].copy()
            
            # Rename columns for consistency
            df.columns = ['ticker', 'company_name', 'exchange', 'category', 'country']
            
            # Remove rows with missing ticker or company name
            df = df.dropna(subset=['ticker', 'company_name'])
            
            # Remove any rows where ticker or company_name are empty strings
            df = df[(df['ticker'].str.strip() != '') & (df['company_name'].str.strip() != '')]
            
            self.logger.info(f"Loaded {len(df)} stock entries")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading stock data: {str(e)}")
            raise
    
    def load_patent_data(self, file_path):
        """
        Load patent applicant data from CSV file
        
        Args:
            file_path (Path): Path to the CSV file
            
        Returns:
            pd.DataFrame: Patent applicant data
        """
        try:
            df = pd.read_csv(file_path)

            #Change the columns name
            df = df.rename(columns={"Company": "company", "Patents_Applied_2010": "y2010", "Patents_Applied_2011": "y2011", "Patents_Applied_2012": "y2012", "Patents_Applied_2013": "y2013" })
            
            # Calculate total patents for sorting
            year_cols = [col for col in df.columns if col.startswith('y')]
            df['total_patents'] = df[year_cols].sum(axis=1)
            
            # Sort by total patents (descending)
            df = df.sort_values('total_patents', ascending=False)
            
            # Clean up company names
            df['company'] = df['company'].str.strip()
            
            # Remove any rows with missing company names
            df = df.dropna(subset=['company'])
            df = df[df['company'].str.strip() != '']
            
            self.logger.info(f"Loaded {len(df)} patent applicants")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading patent data: {str(e)}")
            raise
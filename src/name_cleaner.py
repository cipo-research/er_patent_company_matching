# src/name_cleaner.py
"""
name cleaning utilities with industry-specific mappings and sector awareness
"""

import re
import unicodedata
import pandas as pd
import numpy as np
import logging
from functools import lru_cache

class NameCleaner:
    """ class to clean and standardize company names with industry awareness"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        #  business entity suffixes
        self.business_suffixes = [
            'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited',
            'llc', 'llp', 'lp', 'co', 'company', 'group', 'holdings',
            'plc', 'ag', 'gmbh', 'sa', 'nv', 'bv', 'srl', 'spa',
            'kabushiki kaisha', 'kk', 'co ltd', 'pte ltd', 'pvt ltd',
            'technologies', 'technology', 'tech', 'systems', 'solutions',
            'enterprises', 'international', 'intl', 'global', 'worldwide',
            'labs', 'laboratories', 'research', 'institute', 'foundation'
        ]
        
        # Industry-specific abbreviation mappings
        self.tech_abbreviations = {
            'ibm': 'international business machines',
            'hp': 'hewlett packard', 'hpe': 'hewlett packard enterprise',
            'dell emc': 'dell technologies', 'emc': 'dell technologies',
            'vmware': 'vmware', 'citrix': 'citrix systems',
            'msft': 'microsoft', 'microsoft corp': 'microsoft',
            'googl': 'alphabet', 'goog': 'alphabet', 'google': 'alphabet',
            'aapl': 'apple', 'apple inc': 'apple',
            'amzn': 'amazon', 'amazon com': 'amazon',
            'meta': 'meta platforms', 'fb': 'meta platforms', 'facebook': 'meta platforms',
            'nflx': 'netflix', 'tsla': 'tesla',
            'nvda': 'nvidia', 'amd': 'advanced micro devices',
            'intc': 'intel', 'qcom': 'qualcomm', 'avgo': 'broadcom',
            'orcl': 'oracle', 'crm': 'salesforce', 'adbe': 'adobe',
            'pypl': 'paypal', 'sq': 'square', 'shop': 'shopify',
            'zoom': 'zoom video communications', 'docn': 'digitalocean',
            'snow': 'snowflake', 'pltr': 'palantir', 'crwd': 'crowdstrike'
        }
        
        self.pharma_biotech_abbreviations = {
            'jnj': 'johnson johnson', 'pfe': 'pfizer', 'mrk': 'merck',
            'abbv': 'abbvie', 'bmy': 'bristol myers squibb',
            'lly': 'eli lilly', 'gild': 'gilead sciences',
            'amgn': 'amgen', 'biib': 'biogen', 'vrtx': 'vertex pharmaceuticals',
            'regn': 'regeneron pharmaceuticals', 'celg': 'celgene',
            'ilmn': 'illumina', 'isrg': 'intuitive surgical',
            'dhr': 'danaher', 'tmo': 'thermo fisher scientific',
            'bax': 'baxter international', 'bdi': 'becton dickinson',
            'abt': 'abbott laboratories', 'mdt': 'medtronic'
        }
        
        self.automotive_abbreviations = {
            'gm': 'general motors', 'f': 'ford motor', 'fcau': 'fiat chrysler',
            'tsla': 'tesla', 'tm': 'toyota motor', 'hmc': 'honda motor',
            'nsany': 'nissan motor', 'bmwyy': 'bmw group',
            'vwagy': 'volkswagen group', 'mbgaf': 'mercedes benz group',
            'race': 'ferrari', 'an': 'autonation'
        }
        
        self.telecom_abbreviations = {
            'att': 'at&t', 't': 'at&t', 'vz': 'verizon communications',
            'tmus': 't mobile us', 'sprint': 't mobile us',
            'dish': 'dish network', 'chtr': 'charter communications',
            'cmcsa': 'comcast', 'ctsh': 'cognizant technology solutions'
        }
        
        self.financial_abbreviations = {
            'jpm': 'jpmorgan chase', 'bac': 'bank of america',
            'wfc': 'wells fargo', 'c': 'citigroup',
            'gs': 'goldman sachs group', 'ms': 'morgan stanley',
            'axp': 'american express', 'v': 'visa', 'ma': 'mastercard',
            'pypl': 'paypal holdings', 'sq': 'block'
        }
        
        self.energy_abbreviations = {
            'xom': 'exxon mobil', 'cvx': 'chevron',
            'cop': 'conocophillips', 'slb': 'schlumberger',
            'hal': 'halliburton', 'bkr': 'baker hughes',
            'oxy': 'occidental petroleum', 'mro': 'marathon oil'
        }
        
        self.industrial_abbreviations = {
            'ge': 'general electric', 'mmm': '3m', 'cat': 'caterpillar',
            'ba': 'boeing', 'lmt': 'lockheed martin',
            'rtx': 'raytheon technologies', 'utx': 'raytheon technologies',
            'hon': 'honeywell international', 'emr': 'emerson electric',
            'itt': 'itt', 'pph': 'vwr international'
        }
        
        # Combine all abbreviations
        self.abbreviation_map = {
            **self.tech_abbreviations,
            **self.pharma_biotech_abbreviations,
            **self.automotive_abbreviations,
            **self.telecom_abbreviations,
            **self.financial_abbreviations,
            **self.energy_abbreviations,
            **self.industrial_abbreviations
        }
        
        # Parent-subsidiary relationships
        self.parent_subsidiary_map = {
            'alphabet': ['google', 'youtube', 'waymo', 'deepmind', 'verily'],
            'berkshire hathaway': ['geico', 'duracell', 'dairy queen'],
            'johnson johnson': ['janssen', 'ethicon', 'depuy synthes'],
            'procter gamble': ['gillette', 'oral b', 'braun', 'pampers'],
            'unilever': ['dove', 'lipton', 'knorr', 'hellmanns'],
            'nestle': ['nespresso', 'kitkat', 'smarties', 'butterfinger'],
            'coca cola': ['sprite', 'fanta', 'minute maid', 'powerade'],
            'pepsico': ['pepsi', 'mountain dew', 'gatorade', 'quaker'],
            'general electric': ['ge healthcare', 'ge aviation', 'ge power'],
            'siemens': ['siemens healthineers', 'siemens mobility', 'siemens energy'],
            'philips': ['philips healthcare', 'philips lighting'],
            'samsung': ['samsung electronics', 'samsung sdi', 'samsung biologics'],
            '3m': ['scotch', 'post it', 'nexcare'],
            'danaher': ['beckman coulter', 'leica microsystems', 'cytiva'],
            'thermo fisher scientific': ['thermo scientific', 'applied biosystems', 'invitrogen']
        }
        
        # Sector mapping for matching
        self.sector_keywords = {
            'technology': ['software', 'tech', 'computing', 'internet', 'cloud', 'digital', 'cyber', 'ai', 'machine learning'],
            'healthcare': ['pharma', 'biotech', 'medical', 'health', 'hospital', 'drug', 'therapeutic', 'diagnostic'],
            'financial': ['bank', 'finance', 'investment', 'insurance', 'credit', 'payment', 'trading'],
            'industrial': ['manufacturing', 'aerospace', 'defense', 'construction', 'machinery', 'equipment'],
            'energy': ['oil', 'gas', 'energy', 'petroleum', 'renewable', 'solar', 'wind'],
            'automotive': ['auto', 'motor', 'vehicle', 'car', 'truck', 'transportation'],
            'telecommunications': ['telecom', 'wireless', 'communications', 'network', 'mobile'],
            'consumer': ['retail', 'consumer', 'food', 'beverage', 'apparel', 'entertainment'],
            'materials': ['chemical', 'steel', 'mining', 'paper', 'packaging'],
            'utilities': ['electric', 'utility', 'power', 'water', 'gas utility']
        }
        
        # Pre-compile regex patterns for performance
        self._compile_patterns()
        
        # Create reverse mapping for faster parent lookups
        self._create_reverse_mappings()
        
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        # Sort suffixes by length (longest first) to avoid partial matches
        self.sorted_suffixes = sorted(self.business_suffixes, key=len, reverse=True)
        
        # Create a single pattern for all suffixes
        suffix_pattern = r'\b(' + '|'.join(re.escape(suffix) for suffix in self.sorted_suffixes) + r')\b'
        self.suffix_regex = re.compile(suffix_pattern, re.IGNORECASE)
        
        # Pre-compile special character patterns
        self.special_char_regex = re.compile(r'[^\w\s]')
        self.whitespace_regex = re.compile(r'\s+')
        
    def _create_reverse_mappings(self):
        """Create reverse mappings for O(1) lookups"""
        self.subsidiary_to_parent = {}
        for parent, subsidiaries in self.parent_subsidiary_map.items():
            for sub in subsidiaries:
                self.subsidiary_to_parent[sub.lower()] = parent
                
        # Create sector keyword set for faster lookups
        self.sector_keyword_sets = {}
        for sector, keywords in self.sector_keywords.items():
            self.sector_keyword_sets[sector] = set(keywords)
    
    @lru_cache(maxsize=10000)
    def detect_sector(self, company_name, category=None):
        """
        Detect company sector based on name and category
        
        Args:
            company_name (str): Company name
            category (str): Optional category from stock data
            
        Returns:
            str: Detected sector
        """
        if pd.isna(company_name) or company_name == '':
            return 'other'
            
        name_lower = str(company_name).lower()
        
        # Handle category safely - check for NaN and convert to string
        category_lower = ''
        if category is not None and not pd.isna(category):
            category_lower = str(category).lower()
        
        # Check category first if available
        if category_lower:
            for sector, keyword_set in self.sector_keyword_sets.items():
                if any(keyword in category_lower for keyword in keyword_set):
                    return sector
        
        # Check company name
        for sector, keyword_set in self.sector_keyword_sets.items():
            if any(keyword in name_lower for keyword in keyword_set):
                return sector
        
        return 'other'
    
    @lru_cache(maxsize=10000)
    def find_parent_company(self, company_name):
        """
        Find parent company if the given name is a subsidiary
        
        Args:
            company_name (str): Company name to check
            
        Returns:
            str: Parent company name or original name if no parent found
        """
        if pd.isna(company_name) or company_name == '':
            return company_name
            
        name_lower = str(company_name).lower()
        
        # Check direct mapping first
        if name_lower in self.subsidiary_to_parent:
            return self.subsidiary_to_parent[name_lower]
        
        # Check partial matches
        for sub, parent in self.subsidiary_to_parent.items():
            if sub in name_lower:
                return parent
        
        return company_name
    
    @lru_cache(maxsize=10000)
    def clean_text(self, text):
        """ text cleaning with better handling of special cases"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower().strip()
        
        # Handle special characters that should be preserved
        text = text.replace('&', ' and ')
        text = text.replace('+', ' plus ')
        text = text.replace('@', ' at ')
        
        # Remove accented characters
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        # Remove special characters but preserve spaces and alphanumeric
        text = self.special_char_regex.sub(' ', text)
        text = self.whitespace_regex.sub(' ', text).strip()
        
        return text
    
    @lru_cache(maxsize=10000)
    def remove_business_suffixes(self, text):
        """ suffix removal with better word boundary detection"""
        if not text:
            return text
        
        # Use pre-compiled regex for all suffixes at once
        text = self.suffix_regex.sub('', text)
        
        # Clean up extra whitespace
        text = self.whitespace_regex.sub(' ', text).strip()
        
        return text
    
    @lru_cache(maxsize=10000)
    def apply_abbreviation_mapping(self, text):
        """abbreviation mapping with partial matching"""
        if not text:
            return text
        
        # Check exact match first (fastest)
        if text in self.abbreviation_map:
            return self.abbreviation_map[text]
        
        # Sort by length (longest first) for better matching
        sorted_abbrevs = sorted(self.abbreviation_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        for abbrev, full_form in sorted_abbrevs:
            # Partial match with word boundaries
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
        
        return text
    
    def clean_stock_names(self, df):
        """ stock name cleaning with sector detection"""
        df = df.copy()
        
        # Vectorized cleaning operations
        self.logger.info("Cleaning company names...")
        
        # Apply cleaning functions using vectorized string operations where possible
        df['company_name_cleaned'] = df['company_name'].fillna('').astype(str).str.lower().str.strip()
        
        # Apply transformations in batches for better cache utilization
        df['company_name_cleaned'] = df['company_name_cleaned'].apply(self.clean_text)
        df['company_name_cleaned'] = df['company_name_cleaned'].apply(self.remove_business_suffixes)
        df['company_name_cleaned'] = df['company_name_cleaned'].apply(self.apply_abbreviation_mapping)
        
        # Detect sectors - handle NaN values in category column
        # Convert to tuple for caching
        df['detected_sector'] = df.apply(
            lambda row: self.detect_sector(row['company_name'], row.get('category')), 
            axis=1
        )
        
        # Find parent companies
        df['parent_company'] = df['company_name_cleaned'].apply(self.find_parent_company)
        
        # Store original name
        df['company_name_original'] = df['company_name']
        
        self.logger.info("cleaning of stock company names completed")
        
        return df
    
    def clean_patent_names(self, df):
        """patent name cleaning with sector detection"""
        df = df.copy()
        
        # Vectorized cleaning operations
        self.logger.info("Cleaning patent applicant names...")
        
        # Apply cleaning functions
        df['company_cleaned'] = df['company'].fillna('').astype(str).str.lower().str.strip()
        df['company_cleaned'] = df['company_cleaned'].apply(self.clean_text)
        df['company_cleaned'] = df['company_cleaned'].apply(self.remove_business_suffixes)
        df['company_cleaned'] = df['company_cleaned'].apply(self.apply_abbreviation_mapping)
        
        # Detect sectors based on company name
        df['detected_sector'] = df['company_cleaned'].apply(
            lambda name: self.detect_sector(name)
        )
        
        # Find parent companies
        df['parent_company'] = df['company_cleaned'].apply(self.find_parent_company)
        
        # Store original name
        df['company_original'] = df['company']
        
        self.logger.info("cleaning of patent applicant names completed")
        
        return df
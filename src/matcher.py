# src/matcher.py
"""
Company matching with sector-based matching, confidence intervals, and parent/subsidiary handling
Enhanced with false positive detection and token-based matching
"""

import pandas as pd
import numpy as np
import logging
from difflib import SequenceMatcher
import jellyfish
from scipy import stats
from functools import lru_cache
import multiprocessing as mp
from collections import defaultdict

class CompanyMatcher:
    """Class to match patent applicants with stock companies"""
    
    def __init__(self, exact_threshold=1.0, fuzzy_threshold=0.85, sector_boost=0.05):
        self.logger = logging.getLogger(__name__)
        self.exact_threshold = exact_threshold
        self.fuzzy_threshold = fuzzy_threshold  # Raised from 0.80
        self.sector_boost = sector_boost  # Reduced from 0.1
        
        # Cache for similarity calculations
        self._similarity_cache = {}
        
        # Common words that shouldn't be primary match indicators
        self.common_terms = {
            'corporation', 'incorporated', 'limited', 'company', 'industries',
            'international', 'technologies', 'systems', 'solutions', 'services',
            'enterprises', 'holdings', 'group', 'energy', 'pharmaceutical',
            'medical', 'healthcare', 'financial', 'capital', 'electronics',
            'global', 'research', 'development', 'manufacturing', 'products'
        }
        
        # Known false positive patterns
        self.false_positive_pairs = {
            ('halliburton', 'aly'),
            ('hoffmann', 'ford'),
            ('cargill', 'carnival'),
            ('dow corning', 'corning'),
            ('toray', 'cca'),
            ('lg electronics', 'pulse electronics'),
            ('merck patent', 'merck'),
            ('nissan', 'honda'),
            ('aircelle', 'aircastle'),
            ('genentech', 'green envirotech'),
        }
    
    @lru_cache(maxsize=100000)
    def _cached_jaro_winkler(self, str1, str2):
        """Cached Jaro-Winkler similarity"""
        return jellyfish.jaro_winkler_similarity(str1, str2)
    
    @lru_cache(maxsize=100000)
    def _cached_levenshtein(self, str1, str2):
        """Cached Levenshtein distance"""
        return jellyfish.levenshtein_distance(str1, str2)
    
    def _extract_core_name(self, name):
        """Extract core company name by removing common terms"""
        tokens = name.lower().split()
        core_tokens = [t for t in tokens if t not in self.common_terms and len(t) > 2]
        return ' '.join(core_tokens) if core_tokens else name.lower()
    
    def is_likely_false_positive(self, name1, name2, sim_score):
        """Check if a match is likely a false positive"""
        
        # Check known false positive pairs
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        for fp1, fp2 in self.false_positive_pairs:
            if (fp1 in name1_lower and fp2 in name2_lower) or \
               (fp2 in name1_lower and fp1 in name2_lower):
                return True
        
        # Extract core company names (remove common terms)
        core1 = self._extract_core_name(name1)
        core2 = self._extract_core_name(name2)
        
        # If core names are very different, it's likely a false positive
        if core1 and core2:
            core_sim = SequenceMatcher(None, core1, core2).ratio()
            if core_sim < 0.5 and sim_score < 0.92:  # High threshold for different core names
                return True
        
        # Check for single letter matches (like "F." matching "Ford")
        tokens1 = name1.split()
        tokens2 = name2.split()
        
        # If one company has a single letter abbreviation
        if any(len(t.strip('.')) == 1 for t in tokens1) or \
           any(len(t.strip('.')) == 1 for t in tokens2):
            # Require higher similarity
            if sim_score < 0.95:
                return True
        
        # Length ratio check - very different lengths often indicate false positives
        len_ratio = min(len(name1), len(name2)) / max(len(name1), len(name2))
        if len_ratio < 0.4 and sim_score < 0.93:
            return True
        
        return False
    
    def calculate_token_similarity(self, name1, name2):
        """Calculate similarity based on important tokens"""
        tokens1 = set(name1.lower().split())
        tokens2 = set(name2.lower().split())
        
        # Remove common terms
        important1 = tokens1 - self.common_terms
        important2 = tokens2 - self.common_terms
        
        if not important1 or not important2:
            return 0.0
        
        # Jaccard similarity on important tokens
        intersection = important1 & important2
        union = important1 | important2
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_similarity(self, str1, str2, sector1=None, sector2=None, parent1=None, parent2=None):
        """
        Enhanced similarity calculation with false positive detection
        
        Args:
            str1, str2 (str): Strings to compare
            sector1, sector2 (str): Sectors of the companies
            parent1, parent2 (str): Parent companies
            
        Returns:
            dict: similarity metrics
        """
        if not str1 or not str2:
            return {'base_score': 0.0, 'final_score': 0.0, 'sector_match': False, 'parent_match': False}
        
        # Create cache key
        cache_key = (str1, str2, sector1, sector2, parent1, parent2)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Early termination for very different lengths
        len_ratio = min(len(str1), len(str2)) / max(len(str1), len(str2))
        if len_ratio < 0.3:  # More strict than before
            result = {'base_score': 0.0, 'final_score': 0.0, 'sector_match': False, 'parent_match': False}
            self._similarity_cache[cache_key] = result
            return result
        
        # Base similarity scores
        seq_sim = SequenceMatcher(None, str1, str2).ratio()
        
        # Early termination if sequence similarity is very low
        if seq_sim < 0.4:  # Raised from 0.3
            result = {'base_score': seq_sim, 'final_score': seq_sim, 'sector_match': False, 'parent_match': False}
            self._similarity_cache[cache_key] = result
            return result
        
        jaro_sim = self._cached_jaro_winkler(str1, str2)
        
        # Levenshtein distance normalized
        lev_distance = self._cached_levenshtein(str1, str2)
        max_len = max(len(str1), len(str2))
        lev_sim = 1 - (lev_distance / max_len) if max_len > 0 else 0
        
        # Token-based similarity (new)
        token_sim = self.calculate_token_similarity(str1, str2)
        
        # Weighted average with token similarity
        # Give more weight to token similarity for longer names
        if max_len > 20:
            base_score = (seq_sim * 0.3 + jaro_sim * 0.3 + lev_sim * 0.15 + token_sim * 0.25)
        else:
            base_score = (seq_sim * 0.4 + jaro_sim * 0.4 + lev_sim * 0.2)
        
        # Check for likely false positives
        if self.is_likely_false_positive(str1, str2, base_score):
            base_score *= 0.7  # Significant penalty for suspected false positives
        
        # Sector matching bonus (reduced from 0.1 to 0.05)
        sector_match = False
        if sector1 and sector2 and sector1 == sector2 and sector1 != 'other':
            sector_match = True
            base_score += self.sector_boost
        
        # Sector mismatch penalty (new)
        elif sector1 and sector2 and sector1 != sector2 and sector1 != 'other' and sector2 != 'other':
            base_score *= 0.95  # Small penalty for different sectors
        
        # Parent company matching
        parent_match = False
        if parent1 and parent2:
            if parent1 == parent2 or str1 in parent2 or str2 in parent1:
                parent_match = True
                base_score += 0.15  # Significant boost for parent-subsidiary relationships
        
        # Ensure score doesn't exceed 1.0
        final_score = min(base_score, 1.0)
        
        result = {
            'base_score': base_score,
            'final_score': final_score,
            'sector_match': sector_match,
            'parent_match': parent_match,
            'seq_sim': seq_sim,
            'jaro_sim': jaro_sim,
            'lev_sim': lev_sim,
            'token_sim': token_sim  # Add token similarity to results
        }
        
        self._similarity_cache[cache_key] = result
        return result
    
    def calculate_confidence_interval(self, scores, confidence_level=0.95):
        """
        Calculate confidence interval for match scores
        
        Args:
            scores (list): List of similarity scores
            confidence_level (float): Confidence level (default 95%)
            
        Returns:
            dict: Confidence interval statistics
        """
        if len(scores) < 2:
            return {'mean': scores[0] if scores else 0, 'ci_lower': 0, 'ci_upper': 1, 'std': 0}
        
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)
        n = len(scores)
        
        # Calculate confidence interval using t-distribution
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        margin_of_error = t_value * (std_score / np.sqrt(n))
        
        ci_lower = max(0, mean_score - margin_of_error)
        ci_upper = min(1, mean_score + margin_of_error)
        
        return {
            'mean': mean_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': std_score,
            'margin_of_error': margin_of_error
        }
    
    def _create_blocking_index(self, df, column_name):
        """
        Create blocking index for faster matching
        Groups companies by first letter and length buckets
        """
        blocking_index = defaultdict(list)
        
        # Reset index to ensure we're using sequential indices
        df = df.reset_index(drop=True)
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            name = row[column_name]
            if not name:
                continue
                
            # First letter blocking
            first_letter = name[0].lower() if name else ''
            
            # Length bucket (group by similar lengths)
            length_bucket = len(name) // 5  # 5-character buckets
            
            block_key = (first_letter, length_bucket)
            blocking_index[block_key].append(idx)
        
        return blocking_index
    
    def find_matches(self, patent_df, stock_df, match_type='all'):
        """
        Find matches with all improvements
        
        Args:
            patent_df (pd.DataFrame): Patent applicant data
            stock_df (pd.DataFrame): Stock company data
            match_type (str): Type of matching ('exact', 'fuzzy', 'all')
            
        Returns:
            pd.DataFrame: matches
        """
        matches = []
        total_patents = len(patent_df)
        processed = 0
        
        # Reset indices to ensure consistency
        patent_df = patent_df.reset_index(drop=True)
        stock_df = stock_df.reset_index(drop=True)
        
        # Create blocking index for stock companies
        self.logger.info("Creating blocking index for faster matching...")
        stock_blocking = self._create_blocking_index(stock_df, 'company_name_cleaned')
        
        # Pre-extract stock data as numpy arrays for faster access
        stock_names = stock_df['company_name_cleaned'].values
        stock_sectors = stock_df['detected_sector'].values
        stock_parents = stock_df['parent_company'].values
        stock_originals = stock_df['company_name_original'].values
        stock_tickers = stock_df['ticker'].values
        stock_exchanges = stock_df['exchange'].values
        stock_countries = stock_df['country'].values
        stock_categories = stock_df.get('category', pd.Series([''] * len(stock_df))).values
        
        for patent_idx, patent_row in patent_df.iterrows():
            processed += 1
            
            if processed % 50 == 0 or processed == total_patents:
                self.logger.info(f"Matching progress: {processed}/{total_patents} ({processed/total_patents*100:.1f}%)")
            
            patent_name = patent_row['company_cleaned']
            patent_sector = patent_row.get('detected_sector', 'other')
            patent_parent = patent_row.get('parent_company', patent_name)
            
            if not patent_name:
                continue
            
            # Determine which blocks to search
            first_letter = patent_name[0].lower() if patent_name else ''
            length_bucket = len(patent_name) // 5
            
            # Search in same block and adjacent blocks
            blocks_to_search = [
                (first_letter, length_bucket - 1),
                (first_letter, length_bucket),
                (first_letter, length_bucket + 1)
            ]
            
            candidate_indices = []
            for block_key in blocks_to_search:
                if block_key in stock_blocking:
                    candidate_indices.extend(stock_blocking[block_key])
            
            # If no candidates in blocks, skip
            if not candidate_indices:
                continue
            
            best_matches = []  # Store multiple good matches for confidence calculation
            
            # Only compare with candidates from blocking
            for stock_idx in candidate_indices:
                # Ensure index is within bounds
                if stock_idx >= len(stock_names):
                    continue
                    
                stock_name = stock_names[stock_idx]
                
                if not stock_name:
                    continue
                
                # Quick pre-filter based on name length
                if abs(len(patent_name) - len(stock_name)) > max(len(patent_name), len(stock_name)) * 0.5:
                    continue
                
                stock_sector = stock_sectors[stock_idx]
                stock_parent = stock_parents[stock_idx]
                
                # Calculate similarity
                sim_result = self.calculate_similarity(
                    patent_name, stock_name, 
                    patent_sector, stock_sector,
                    patent_parent, stock_parent
                )
                
                # Early termination if score is too low
                if sim_result['final_score'] < self.fuzzy_threshold:
                    continue
                
                # Check if match meets criteria
                is_exact = sim_result['final_score'] >= self.exact_threshold
                is_fuzzy = sim_result['final_score'] >= self.fuzzy_threshold
                
                if (match_type == 'exact' and is_exact) or \
                   (match_type == 'fuzzy' and is_fuzzy and not is_exact) or \
                   (match_type == 'all' and (is_exact or is_fuzzy)):
                    
                    match_data = {
                        'applicant_name_patstat': patent_row['company_original'],
                        'company_name_stock': stock_originals[stock_idx],
                        'ticker_stock': stock_tickers[stock_idx],
                        'exchange': stock_exchanges[stock_idx],
                        'country': stock_countries[stock_idx],
                        'category': stock_categories[stock_idx],
                        'patent_sector': patent_sector,
                        'stock_sector': stock_sector,
                        'sector_match': sim_result['sector_match'],
                        'parent_match': sim_result['parent_match'],
                        'match_type': 'exact' if is_exact else 'fuzzy',
                        'match_score': sim_result['final_score'],
                        'base_score': sim_result['base_score'],
                        'sequence_similarity': sim_result['seq_sim'],
                        'jaro_similarity': sim_result['jaro_sim'],
                        'levenshtein_similarity': sim_result['lev_sim'],
                        'token_similarity': sim_result.get('token_sim', 0),  # Add token similarity
                        'total_patents': patent_row['total_patents']
                    }
                    
                    best_matches.append((sim_result['final_score'], match_data))
            
            # Keep only the best match(es) per patent applicant
            if best_matches:
                # Sort by score and take the best
                best_matches.sort(key=lambda x: x[0], reverse=True)
                best_score = best_matches[0][0]
                
                # Take all matches within 0.05 of the best score for confidence calculation
                top_matches = [match for score, match in best_matches if score >= best_score - 0.05]
                
                if len(top_matches) > 1:
                    # Calculate confidence interval for multiple good matches
                    scores = [match['match_score'] for match in top_matches]
                    ci_stats = self.calculate_confidence_interval(scores)
                    
                    # Add confidence statistics to the best match
                    best_match = top_matches[0]
                    best_match.update({
                        'confidence_mean': ci_stats['mean'],
                        'confidence_lower': ci_stats['ci_lower'],
                        'confidence_upper': ci_stats['ci_upper'],
                        'confidence_std': ci_stats['std'],
                        'alternative_matches': len(top_matches) - 1
                    })
                else:
                    # Single match - high confidence
                    best_match = top_matches[0]
                    best_match.update({
                        'confidence_mean': best_match['match_score'],
                        'confidence_lower': best_match['match_score'],
                        'confidence_upper': best_match['match_score'],
                        'confidence_std': 0.0,
                        'alternative_matches': 0
                    })
                
                matches.append(best_match)
        
        return pd.DataFrame(matches)
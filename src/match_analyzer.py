# src/match_analyzer.py
"""
Analysis utilities for understanding match quality and patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

class MatchAnalyzer:
    """Class to analyze matching results and provide insights"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_match_quality(self, matches_df):
        """
        Analyze the quality and patterns in matches
        
        Args:
            matches_df (pd.DataFrame): Matches dataframe
            
        Returns:
            dict: Analysis results
        """
        analysis = {}
        
        # Use vectorized operations for better performance
        # Basic statistics
        analysis['total_matches'] = len(matches_df)
        analysis['exact_matches'] = (matches_df['match_type'] == 'exact').sum()
        analysis['fuzzy_matches'] = (matches_df['match_type'] == 'fuzzy').sum()
        
        # Score distribution - using numpy for faster computation
        scores = matches_df['match_score'].values
        analysis['score_stats'] = {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
        
        # Confidence analysis - vectorized operations
        confidence_means = matches_df['confidence_mean'].values
        confidence_lowers = matches_df['confidence_lower'].values
        
        analysis['confidence_stats'] = {
            'mean_confidence': np.mean(confidence_means),
            'high_confidence_count': (confidence_lowers >= 0.85).sum(),
            'medium_confidence_count': ((confidence_lowers >= 0.70) & (confidence_lowers < 0.85)).sum(),
            'low_confidence_count': (confidence_lowers < 0.70).sum()
        }
        
        # Sector analysis - using value_counts which is optimized
        analysis['sector_analysis'] = {
            'sector_matches_count': matches_df['sector_match'].sum(),
            'sector_match_rate': matches_df['sector_match'].mean(),
            'top_sectors': matches_df['patent_sector'].value_counts().head(5).to_dict()
        }
        
        # Parent-subsidiary analysis
        analysis['parent_subsidiary_analysis'] = {
            'parent_matches_count': matches_df['parent_match'].sum(),
            'parent_match_rate': matches_df['parent_match'].mean()
        }
        
        # Patent volume analysis
        total_patents = matches_df['total_patents'].values
        analysis['patent_volume_analysis'] = {
            'total_patents_matched': np.sum(total_patents),
            'avg_patents_per_match': np.mean(total_patents),
            'median_patents_per_match': np.median(total_patents),
            'top_patent_holders': matches_df.nlargest(5, 'total_patents')[
                ['applicant_name_patstat', 'company_name_stock', 'total_patents']
            ].to_dict('records')
        }
        
        return analysis
    
    def generate_match_report(self, matches_df, output_path=None):
        """
        Generate a comprehensive match analysis report
        
        Args:
            matches_df (pd.DataFrame): Matches dataframe
            output_path (str): Optional path to save the report
            
        Returns:
            str: Report text
        """
        analysis = self.analyze_match_quality(matches_df)
        
        report = []
        report.append("="*60)
        report.append("PATENT-STOCK MATCHING ANALYSIS REPORT")
        report.append("="*60)
        
        # Basic Statistics
        report.append(f"\nBASIC STATISTICS")
        report.append(f"Total matches found: {analysis['total_matches']:,}")
        report.append(f"Exact matches: {analysis['exact_matches']:,} ({analysis['exact_matches']/analysis['total_matches']*100:.1f}%)")
        report.append(f"Fuzzy matches: {analysis['fuzzy_matches']:,} ({analysis['fuzzy_matches']/analysis['total_matches']*100:.1f}%)")
        
        # Match Quality
        report.append(f"\nMATCH QUALITY METRICS")
        report.append(f"Average match score: {analysis['score_stats']['mean']:.3f}")
        report.append(f"Median match score: {analysis['score_stats']['median']:.3f}")
        report.append(f"Score standard deviation: {analysis['score_stats']['std']:.3f}")
        report.append(f"Score range: {analysis['score_stats']['min']:.3f} - {analysis['score_stats']['max']:.3f}")
        
        # Confidence Analysis
        report.append(f"\nCONFIDENCE ANALYSIS")
        report.append(f"Average confidence: {analysis['confidence_stats']['mean_confidence']:.3f}")
        report.append(f"High confidence matches (CI >= 0.85): {analysis['confidence_stats']['high_confidence_count']:,}")
        report.append(f"Medium confidence matches (CI 0.70-0.85): {analysis['confidence_stats']['medium_confidence_count']:,}")
        report.append(f"Low confidence matches (CI < 0.70): {analysis['confidence_stats']['low_confidence_count']:,}")
        
        # Sector Analysis
        report.append(f"\nSECTOR ANALYSIS")
        report.append(f"Same-sector matches: {analysis['sector_analysis']['sector_matches_count']:,} ({analysis['sector_analysis']['sector_match_rate']*100:.1f}%)")
        report.append(f"Top sectors by match count:")
        for sector, count in analysis['sector_analysis']['top_sectors'].items():
            report.append(f"  {sector.title()}: {count:,} matches")
        
        # Parent-Subsidiary Analysis
        report.append(f"\nPARENT-SUBSIDIARY ANALYSIS")
        report.append(f"Parent-subsidiary matches: {analysis['parent_subsidiary_analysis']['parent_matches_count']:,} ({analysis['parent_subsidiary_analysis']['parent_match_rate']*100:.1f}%)")
        
        # Patent Volume Analysis
        report.append(f"\nPATENT VOLUME ANALYSIS")
        report.append(f"Total patents covered: {analysis['patent_volume_analysis']['total_patents_matched']:,}")
        report.append(f"Average patents per matched company: {analysis['patent_volume_analysis']['avg_patents_per_match']:.0f}")
        report.append(f"Median patents per matched company: {analysis['patent_volume_analysis']['median_patents_per_match']:.0f}")
        
        report.append(f"\nTOP PATENT HOLDERS (MATCHED)")
        for i, holder in enumerate(analysis['patent_volume_analysis']['top_patent_holders'], 1):
            report.append(f"{i}. {holder['applicant_name_patstat']} -> {holder['company_name_stock']} ({holder['total_patents']:,} patents)")
        
        # Quality Recommendations
        report.append(f"\nQUALITY RECOMMENDATIONS")
        high_conf_rate = analysis['confidence_stats']['high_confidence_count'] / analysis['total_matches']
        if high_conf_rate < 0.5:
            report.append("Consider increasing fuzzy matching threshold for higher confidence matches")
        if analysis['sector_analysis']['sector_match_rate'] < 0.3:
            report.append("Consider improving sector detection algorithms")
        if analysis['parent_subsidiary_analysis']['parent_match_rate'] < 0.1:
            report.append("Consider expanding parent-subsidiary relationship database")
        
        report.append("\n" + "="*60)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Match analysis report saved to {output_path}")
        
        return report_text
    
    def plot_match_distributions(self, matches_df, output_dir=None):
        """
        Create visualization plots for match analysis
        
        Args:
            matches_df (pd.DataFrame): Matches dataframe
            output_dir (str): Directory to save plots
        """
        # Use 'default' style if seaborn-v0_8 is not available
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Create figure with tight layout for better performance
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), tight_layout=True)
        fig.suptitle('Patent-Stock Matching Analysis', fontsize=16, fontweight='bold', y=1.02)
        
        # Pre-compute data for all plots to minimize dataframe operations
        match_scores = matches_df['match_score'].values
        confidence_lowers = matches_df['confidence_lower'].values
        total_patents = matches_df['total_patents'].values
        
        # 1. Match Score Distribution
        axes[0, 0].hist(match_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(match_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(match_scores):.3f}')
        axes[0, 0].set_xlabel('Match Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Match Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence Interval Analysis
        conf_categories = ['High (>=0.85)', 'Medium (0.70-0.85)', 'Low (<0.70)']
        conf_counts = [
            (confidence_lowers >= 0.85).sum(),
            ((confidence_lowers >= 0.70) & (confidence_lowers < 0.85)).sum(),
            (confidence_lowers < 0.70).sum()
        ]
        colors = ['green', 'orange', 'red']
        axes[0, 1].pie(conf_counts, labels=conf_categories, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Confidence Level Distribution')
        
        # 3. Sector Distribution
        top_sectors = matches_df['patent_sector'].value_counts().head(8)
        axes[0, 2].bar(range(len(top_sectors)), top_sectors.values, color='lightcoral')
        axes[0, 2].set_xlabel('Sector')
        axes[0, 2].set_ylabel('Number of Matches')
        axes[0, 2].set_title('Top Sectors by Match Count')
        axes[0, 2].set_xticks(range(len(top_sectors)))
        axes[0, 2].set_xticklabels(top_sectors.index, rotation=45, ha='right')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Match Type Distribution
        match_types = matches_df['match_type'].value_counts()
        axes[1, 0].pie(match_types.values, labels=match_types.index, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Match Type Distribution')
        
        # 5. Patent Volume vs Match Score
        axes[1, 1].scatter(total_patents, match_scores, alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Total Patents')
        axes[1, 1].set_ylabel('Match Score')
        axes[1, 1].set_title('Patent Volume vs Match Score')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Features Impact
        sector_match_counts = [matches_df['sector_match'].sum(), (~matches_df['sector_match']).sum()]
        parent_match_counts = [matches_df['parent_match'].sum(), (~matches_df['parent_match']).sum()]
        
        x = np.arange(2)
        width = 0.35
        
        axes[1, 2].bar(x - width/2, [sector_match_counts[0], parent_match_counts[0]], 
                       width, label='Yes', color='lightgreen')
        axes[1, 2].bar(x + width/2, [sector_match_counts[1], parent_match_counts[1]], 
                       width, label='No', color='lightblue')
        axes[1, 2].set_xlabel('Feature Type')
        axes[1, 2].set_ylabel('Number of Matches')
        axes[1, 2].set_title('Features Impact')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(['Sector Match', 'Parent Match'])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        if output_dir:
            plot_path = f"{output_dir}/matching_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Analysis plots saved to {plot_path}")
        
        plt.show()
        return fig
#!/usr/bin/env python
"""
build_patstat_table1_only.py

Optimized version to build only Table 1: annual applications per company

Outputs:
  data/processed/table1_annual_applications_per_company.csv
"""

import os
import time
import pandas as pd
from collections import defaultdict
from src.config import get_conn

CHUNK_SIZE = 50_000

def chunk_query(conn, base_query: str, chunk_size: int = CHUNK_SIZE):
    """
    Stream the given SQL using keyset pagination on appln_id,
    yielding DataFrame pages of up to chunk_size rows.
    """
    last_id = 0
    while True:
        paged_sql = f"""
        {base_query}
          AND appln_id > {last_id}
        ORDER BY appln_id
        LIMIT {chunk_size}
        """
        page = pd.read_sql_query(paged_sql, conn)
        if page.empty:
            break
        yield page
        last_id = page['appln_id'].max()

def process_applications_streaming(conn):
    """
    Process applications in chunks, filtering and building family mappings on-the-fly.
    Returns: family_data for Table 1 processing
    """
    print("Loading and filtering applications with international ID...")
    
    base_query = """
    SELECT appln_id, appln_filing_year, appln_auth, internat_appln_id
      FROM tls201_appln
     WHERE internat_appln_id IS NOT NULL
       AND appln_filing_year BETWEEN 2010 AND 2013
       AND appln_auth IN ('CA', 'US')
    """
    
    # Track families and their CA/US applications
    family_apps = defaultdict(lambda: {'CA': [], 'US': []})
    family_years = {}
    total_rows = 0
    
    for page in chunk_query(conn, base_query):
        total_rows += len(page)
        print(f"  Processed {total_rows:,} CA/US rows so far...")
        
        # Group by family and collect CA/US applications
        for _, row in page.iterrows():
            fam_id = row['internat_appln_id']
            auth = row['appln_auth']
            app_id = row['appln_id']
            year = row['appln_filing_year']
            
            family_apps[fam_id][auth].append(app_id)
            # Store the year for this family (assuming consistent within family)
            if fam_id not in family_years:
                family_years[fam_id] = year
    
    # Filter to families that have both CA and US applications
    ca_us_families = {
        fam_id: apps for fam_id, apps in family_apps.items()
        if apps['CA'] and apps['US']
    }
    
    print(f"Found {len(ca_us_families):,} families with both CA and US applications")
    
    # Create family data for Table 1 processing
    family_data = []
    for fam_id, apps in ca_us_families.items():
        for app_id in apps['CA'] + apps['US']:
            family_data.append({
                'appln_id': app_id,
                'internat_appln_id': fam_id,
                'appln_filing_year': family_years[fam_id]
            })
    
    return pd.DataFrame(family_data)

def load_applicant_data_chunked(conn, appln_ids):
    """Load lead applicants and company names in chunks"""
    print("Loading lead applicants and company names...")
    
    # Convert to list for chunking
    appln_id_list = list(appln_ids)
    all_merged_data = []
    
    # Process applicant data in chunks to avoid large IN clauses
    for i in range(0, len(appln_id_list), CHUNK_SIZE):
        chunk_ids = appln_id_list[i:i + CHUNK_SIZE]
        id_list = "(" + ",".join(map(str, chunk_ids)) + ")"
        
        # Join leads and persons in a single query for efficiency
        combined_sql = f"""
        SELECT 
            la.appln_id,
            la.person_id,
            COALESCE(p.person_name, 'Unknown Company') as person_name
        FROM tls207_pers_appln la
        LEFT JOIN tls206_person p ON la.person_id = p.person_id
        WHERE la.applt_seq_nr = 1
          AND la.appln_id IN {id_list}
        """
        
        chunk_data = pd.read_sql_query(combined_sql, conn)
        all_merged_data.append(chunk_data)
        
        print(f"  Loaded applicant data for {i + len(chunk_ids):,}/{len(appln_id_list):,} applications")
    
    if all_merged_data:
        merged_applicants = pd.concat(all_merged_data, ignore_index=True)
        print(f" Total applicant records: {len(merged_applicants):,}")
        return merged_applicants
    else:
        return pd.DataFrame(columns=['appln_id', 'person_id', 'person_name'])

def build_table1(family_data, applicant_data):
    """Build Table 1: distinct families per company/year"""
    print("Building Table 1 (annual counts per company)...")
    
    if applicant_data.empty:
        return pd.DataFrame(columns=[
            'Company',
            'Patents_Applied_2010', 'Patents_Applied_2011',
            'Patents_Applied_2012', 'Patents_Applied_2013'
        ])
    
    # Merge family data with applicant data
    merged = family_data.merge(applicant_data, on='appln_id', how='left')
    merged['person_name'] = merged['person_name'].fillna('Unknown Company')
    
    # Count distinct families per company per year
    counts = (
        merged
        .groupby(['person_name', 'appln_filing_year'])['internat_appln_id']
        .nunique()
        .unstack(fill_value=0)
    )
    
    # Ensure all year columns exist
    year_columns = ['Patents_Applied_2010', 'Patents_Applied_2011', 
                   'Patents_Applied_2012', 'Patents_Applied_2013']
    
    # Rename columns
    counts.columns = [f'Patents_Applied_{int(c)}' for c in counts.columns]
    
    # Add missing year columns with 0s
    for col in year_columns:
        if col not in counts.columns:
            counts[col] = 0
    
    # Select and order columns
    table1 = (
        counts[year_columns]
        .reset_index()
        .rename(columns={'person_name': 'Company'})
    )
    
    print(f"Table 1 created: {len(table1):,} companies")
    return table1

def main():
    start = time.perf_counter()

    # Connect and set schema
    conn = get_conn()
    conn.cursor().execute("SET search_path = patstats, public;")
    print(" Connected to database")

    try:
        # Step 1: Process applications with streaming and filtering
        family_data = process_applications_streaming(conn)
        
        if family_data.empty:
            print("No CA+US joint applications found.")
            return

        # Step 2: Load applicant data in chunks
        all_appln_ids = family_data['appln_id'].unique()
        applicant_data = load_applicant_data_chunked(conn, all_appln_ids)

        # Step 3: Build Table 1
        table1 = build_table1(family_data, applicant_data)

        # Step 4: Save Table 1 to CSV
        print("Saving Table 1...")
        out_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            'data', 'processed'
        )
        os.makedirs(out_dir, exist_ok=True)

        t1_path = os.path.join(out_dir, 'table1_annual_applications_per_company.csv')
        table1.to_csv(t1_path, index=False)
        print(f"   Table 1 saved to {t1_path}")

        elapsed = time.perf_counter() - start
        print(f"Completed in {elapsed:.1f}s")

    finally:
        conn.close()

if __name__ == '__main__':
    main()
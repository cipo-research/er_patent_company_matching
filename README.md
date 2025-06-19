# Patent-Stock Company Matcher


##  Input Data

### Patent Data (CSV file)
```
applicant_name                     total_patents
Samsung Electronics Co., Ltd.       1234
International Business Machines     567
BASF SE                            890
```

### Stock Data (CSV file)
```
company_name      ticker    exchange    sector
Samsung Elec      SSNLF     OTC        Technology
IBM Corp          IBM       NYSE       Technology  
BASF SE           BASFY     OTC        Materials
```

##  How It Works - Simple Explanation

### 1. **Name Cleaning** 
First, we clean up company names to make matching easier:

**Example transformations:**
- "Apple Inc." → "apple"
- "Microsoft Corporation" → "microsoft"
- "AT&T Corp." → "at and t"
- "SAMSUNG ELECTRONICS CO., LTD." → "samsung electronics"

**What gets removed:**
- Business suffixes (Inc, Corp, Ltd, LLC, GmbH, etc.)
- Punctuation and special characters
- Extra spaces
- Converts everything to lowercase

### 2. **Smart Matching Algorithms** 

The tool uses three different methods to compare names:

#### **Sequence Matcher (40% weight)**
Finds the longest matching parts between two names.
```
Example: "sony corporation" vs "sony corp"
Finds: "sony corp" matches perfectly
Score: 0.857 (very high!)
```

#### **Jaro-Winkler (40% weight)**
Good at catching typos and similar-sounding names.
```
Example: "pfizer" vs "phizer" 
Score: 0.867 (catches the typo!)
```

#### **Levenshtein Distance (20% weight)**
Counts how many letter changes needed.
```
Example: "nokia" → "mokia"
Only 1 change needed (n→m)
Score: 0.80 (1 change out of 5 letters)
```

### 3. **Industry Detection** 

The tool identifies what industry each company belongs to by looking for keywords:

**Technology keywords**: software, computer, electronic, semiconductor
```
"Apple Computer Inc" → detected as "technology"
"Advanced Micro Devices" → detected as "technology"
```

**Healthcare keywords**: pharmaceutical, medical, healthcare, drug
```
"Pfizer Pharmaceuticals" → detected as "healthcare"
"Johnson & Johnson Medical" → detected as "healthcare"
```

**Energy keywords**: oil, gas, petroleum, energy
```
"ExxonMobil Oil Company" → detected as "energy"
"Solar Energy Corp" → detected as "energy"
```

### 4. **Parent Company Recognition** 

Recognizes when companies are related:
```
"Google Inc" → parent: "Alphabet"
"WhatsApp Inc" → parent: "Meta"
"LinkedIn Corp" → parent: "Microsoft"
```

### 5. **Smart Scoring System** 

Each match gets a score from 0 to 1:
- **1.00** = Perfect match (e.g., "Apple Inc" ↔ "Apple Inc")
- **0.95+** = Excellent match (e.g., "Microsoft Corp" ↔ "Microsoft Corporation")
- **0.85-0.95** = Good match (e.g., "Sony Corp" ↔ "Sony Corporation Japan")
- **0.80-0.85** = Possible match (needs human review)
- **Below 0.80** = Not considered a match

**Bonus points added for:**
- Same industry (+0.10): "Apple Tech Inc" + "Apple Corporation" [both in technology]
- Parent-subsidiary relationship (+0.15): "YouTube" + "Google" [owned by same parent]

##  Performance Optimizations - maybe batchProcessing?



## Output Explained

The tool creates a CSV file with these columns:

| Column | What it means | Example |
|--------|---------------|---------|
| **applicant_name_patstat** | Original patent filer name | "Samsung Electronics Co., Ltd." |
| **company_name_stock** | Matched stock market name | "Samsung Electronics Co Ltd" |
| **ticker_stock** | Stock symbol | "SSNLF" |
| **match_type** | How well they matched | "exact" or "fuzzy" |
| **match_score** | Similarity score (0-1) | 0.95 |
| **total_patents** | Number of patents filed | 1234 |
| **sector_match** | Same industry? | TRUE/FALSE |
| **confidence_mean** | How confident in the match | 0.95 |
| **alternative_matches** | Other possible matches found | 2 |

##  Real Examples from the System

### Perfect Matches (Score = 1.00)
- **BASF SE** ↔ **BASF SE** 
- **Sony Corporation** ↔ **Sony Corporation** 
- **Honda Motor Co., Ltd.** ↔ **Honda Motor Co., Ltd.** 

### Great Fuzzy Matches (Score > 0.90)
- **Telefonaktiebolaget L M Ericsson (Publ)** ↔ **Telefonaktiebolaget LM Ericsson (publ)** (0.976)
  - *Caught the minor differences in capitalization and punctuation!*

- **Bayer Pharma Aktiengesellschaft** ↔ **Bayer Aktiengesellschaft** (0.963)
  - *Recognized these as related despite "Pharma" difference*

### Good Matches Needing Review (Score 0.80-0.90)
- **Halliburton Energy Services, Inc.** ↔ **Aly Energy Services, Inc.** (0.850)
  - *Both are energy services but different companies - human review needed!*

### Why Some Matches Are Tricky
- **F. Hoffmann-La Roche AG** matched with **Ford Motor Company** (0.88)
  - *The "F." confused the system - but sector mismatch flagged it!*

##  How to Use

1. **Prepare your data files:**
   - Patent data CSV with company names
   - Stock data CSV with company names

2. **Run the matcher:**
   ```bash
   python src/main.py
   ```

3. **Check the results:**
   - Look for matches with confidence > 0.95 first
   - Review matches between 0.85-0.95
   - Check sector mismatches

##  Tips for Best Results

1. **High patent count companies match better** - they're usually large, well-known companies
2. **Pharmaceutical companies match very well** - consistent naming conventions
3. **Check alternative_matches > 0** - means multiple possible matches found
4. **Sector mismatch + high score** = verify manually, might be a conglomerate

##  Performance Stats

- **Top 1000 patent applicants**: ~5 minutes
- **Full dataset (556K applicants)**: 15-30 hours
- **Success rate**: ~80% of Fortune 500 companies matched correctly


##  Quality Metrics

The system provides confidence scores for each match:
- **confidence_mean**: Average confidence across similar matches
- **confidence_lower/upper**: Statistical confidence interval
- **alternative_matches**: Number of other potential matches


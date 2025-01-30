# Nabtah 🌿

This project analyzes a conversation to recommend a high school track based on pattern matching, sentiment analysis, Career paths and Holland Codes

## Features
SpaCy Matcher to detect relevant career-related and Holland codes (RIASEC) keywords in a given conversation
BERT-based sentiment analysis to determine the sentiment towards specific tracks and RIASEC codes
Scoring system to evaluate and recommend one of the five tracks:
  - Computer Science and Engineering Track
  - Health and Life Sciences Track
  - Business Administration Track
  - Sharia (Islamic Law) Track
  - General Track
Radar Chart Visualization to compare different track scores

## How It Works

###1. Pattern Matching with SpaCy
  - Loads keyword patterns from patterns.json
  - Uses SpaCy's Matcher to find relevant keywords in a given conversation

2. Sentiment Analysis using BERT
  - Extracts context words around the matched keywords
  - Uses the sentiment model to classify sentiment into:
    - Positive: +1 to the track/RIASEC code score
    - Negative: -1 to the track/RIASEC code score
   
3. RIASEC Code Analysis
  - Maps tracks to RIASEC codes using track_to_RIASEC.json

4. Track Recommendation
  - The total score for each track is computed as:
      $\text{Total Score} = \text{Track Score} + 0.5 \times \sum (\text{RIASEC Scores})$
  - The track with the highest score is recommended

5. Radar Chart Visualization
  - Generates a Radar Chart to compare different tracks score

import spacy
from spacy.matcher import Matcher
from transformers import pipeline
import torch
import json # to read the patterns json file
import plotly.graph_objects as go # for Radar Chart


# loading the patterns from patterns.json file
with open("Data/patterns.json", 'r') as patterns_json:
    patterns = json.load(patterns_json)

nlp = spacy.load('en_core_web_lg')

# BERT
sentiment_analyzer = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

# Matcher 
matcher = Matcher(nlp.vocab) # Initializing the Matcher

for pattern_label, pattern in patterns.items():
    matcher.add(pattern_label, pattern)

# %%

# Apply matcher with BERT sentiment analysis
class Analyze:

    def __init__(self):
        self.track_scores = {}
        self.RIASEC_scores = {}
        self.total_score = {}

    def analyze_conversation(self, conversation):
        doc = nlp(conversation)
        matches = matcher(doc)

        # Tracks scores
        track_scores = {
            "General": 0,
            "Computer Science and Engineering": 0,
            "Health and Life Sciences": 0,
            "Business Administration": 0,
            "Sharia": 0,

            "Realistic": 0,
            "Investigative": 0,
            "Artistic": 0,
            "Social": 0,
            "Enterprising": 0,
            "Conventional": 0
        }

        for match_id, start, end in matches:
            # matched track label
            track_label = nlp.vocab.strings[match_id]

            # Look at 3 words before and after the keyword for sentiment analysis
            keyword_context = doc[max(0, start-3):min(len(doc), end+3)]
            context_text = keyword_context.text

            # BERT based sentiment analysis on the context
            sentiment_result = sentiment_analyzer(context_text)[0]  # sentiment result

            # Scoring based on BERT sentiment label
            if sentiment_result['label'] in ['4 stars', '5 stars']:  # Positive sentiment
                track_scores[track_label] += 1
            elif sentiment_result['label'] in ['1 star', '2 stars']:  # Negative sentiment
                track_scores[track_label] -= 1
            else:  
                track_scores[track_label] += 1  # neutral as positive

        track_scores = list(track_scores.items())
        return dict(track_scores[:5]), dict(track_scores[5:]) # (High school tracks score, RIASEC codes score)    

    def analyze(self, conversation):

        self.track_scores, self.RIASEC_scores = self.analyze_conversation(conversation)

        if self.track_scores[max(self.track_scores, key=self.track_scores.get)] + self.track_scores[min(self.track_scores, key=self.track_scores.get)] == 0:
            return {"status": "warning", "message": "It seems the assessment was not completed. Please retake the assessment"}
        if self.RIASEC_scores[max(self.RIASEC_scores, key=self.RIASEC_scores.get)] + self.RIASEC_scores[min(self.RIASEC_scores, key=self.RIASEC_scores.get)] == 0:
            return {"status": "warning", "message": "It seems the assessment was not completed. Please retake the assessment"}

        # Calculating the total score
        with open("Data/track_to_RIASEC.json", "r") as track_to_RIASEC_json:
            track_to_RIASEC = json.load(track_to_RIASEC_json)

        for track, RIASEC_Codes in track_to_RIASEC.items():
            self.total_score[track] = self.track_scores[track] + 0.5 * sum(self.RIASEC_scores[RIASEC_Code] for RIASEC_Code in RIASEC_Codes) # 50% of RAISEC score affect the total score

        # returning recommended track
        max_track = max(self.total_score, key=self.total_score.get)
        return {"status": "success", "message": "Recommended Track: " + max_track}

    # Radar Chart
    def radarChart(self, RIASEC=False, original=False, normalize=True): 
        
        scores = self.total_score
        title='High School Track Recommendation'

        if RIASEC == True:
            scores = self.RIASEC_scores
            title = 'Holland Codes Assessment'
        if original == True:
            scores = self.track_scores
            title='Original Scores'

        # converting the scores into lists so we can sort them for better appearance
        tracks = list(scores.keys())
        scores = list(scores.values())
        sorted_data = sorted(zip(scores, tracks), reverse=True)
        sorted_scores, sorted_tracks = zip(*sorted_data)

        if normalize:
            min_score = min(sorted_scores)
            max_score = max(sorted_scores)
            print("max = ", max_score)
            print("min = ", min_score)
            scores = [(score - min_score) / (max_score - min_score) for score in sorted_scores]
            scores = [round(score, 3) for score in scores] # ".3f"

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
            r=scores,
            theta=sorted_tracks,
            fill='toself',
            name=title,
            fillcolor='#a8d4b4',
            line_color="#0e6926",
            opacity=0.5
        )
        )


        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1] if normalize else [0, max(scores)]
                )),
            title=title+" Radar Chart",
            showlegend=False
        )

        return fig

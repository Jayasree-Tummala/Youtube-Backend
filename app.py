import torch
import uvicorn
import pandas as pd
import os
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from googleapiclient.discovery import build
from fastapi.requests import Request

# Initialize FastAPI
app = FastAPI()

# Setup templates & static files
app.mount("/static", StaticFiles(directory="plots"), name="static")
templates = Jinja2Templates(directory="templates")

# 🔑 API Key & Setup
KEY = "AIzaSyCpUQOmSk7ULuXRMyav7WGw0mcndwqJ-VQ"
youtube = build("youtube", "v3", developerKey=KEY)

# 🔍 Load Models
sentiment_model_path = "Jt182001/sentiment_analysis_model"
narcissism_model_path = "Jt182001/youtube_narcissism_model"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
narcissism_model = AutoModelForSequenceClassification.from_pretrained(narcissism_model_path)
narcissism_tokenizer = AutoTokenizer.from_pretrained(narcissism_model_path)

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# 📂 Create a directory for saving images if not exists
if not os.path.exists("plots"):
    os.makedirs("plots")


# 🧠 Sentiment Analysis Function
def get_sentiment(sentence):
    inputs = sentiment_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    predicted_label = torch.argmax(scores).item()

    vader_score = analyzer.polarity_scores(sentence)['compound']
    if -0.5 <= vader_score <= 0.5:
        return "Neutral"
    else:
        return "Positive" if predicted_label == 1 else "Negative"


# 🧠 Narcissism Analysis Function
def check_narcissism(sentence):
    inputs = narcissism_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = narcissism_model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    predicted_label = torch.argmax(scores).item()

    return "Yes" if predicted_label == 1 else "No"


# 🛠 Fetch YouTube Comments
def get_youtube_comments(video_id, max_results=100):
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, textFormat="plainText", maxResults=max_results
    )
    response = request.execute()

    comments = []
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments


# 🎨 Generate WordCloud
def generate_wordcloud(text, save_path="plots/wordcloud.png"):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud of YouTube Comments")
    plt.savefig(save_path)
    plt.close()


# 📊 Generate Sentiment Analysis Graph
def generate_sentiment_plot(df, save_path="plots/sentiment_plot.png"):
    sentiment_counts = df["Sentiment"].value_counts()

    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind="bar", color=["green", "gray", "red"])
    plt.title("Sentiment Analysis")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig(save_path)
    plt.close()


# 📊 Generate Narcissism Analysis Graph
def generate_narcissism_plot(df, save_path="plots/narcissism_plot.png"):
    narcissism_counts = df[df["Sentiment"] == "Negative"]["Narcissism"].value_counts()

    plt.figure(figsize=(6, 4))
    narcissism_counts.plot(kind="bar", color=["red", "blue"])
    plt.title("Narcissistic Comments (Only from Negative)")
    plt.xlabel("Narcissism")
    plt.ylabel("Count")
    plt.savefig(save_path)
    plt.close()





# 🎯 API Endpoint - Analyze Video Comments
@app.get("/analyze")
def analyze_video(video_id: str):
    comments = get_youtube_comments(video_id)
    results = []

    for comment in comments:
        sentiment = get_sentiment(comment)
        narcissism = "Not Checked" if sentiment != "Negative" else check_narcissism(comment)
        results.append({"Sentence": comment, "Sentiment": sentiment, "Narcissism": narcissism})

    df = pd.DataFrame(results)

    # Generate Graphs
    all_comments_text = " ".join(df["Sentence"])
    generate_wordcloud(all_comments_text)
    generate_sentiment_plot(df)
    generate_narcissism_plot(df)

    return {"video_id": video_id, "analysis": results}


# 🎨 Dashboard Page (Single Page with All Graphs)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
  
    return templates.TemplateResponse("index.html", {"request": request})


# 🚀 Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

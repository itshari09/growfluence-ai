from textblob import TextBlob

def analyze_caption(text):
    blob = TextBlob(text)

    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        mood = "Positive 😊"
    elif sentiment < 0:
        mood = "Negative 😡"
    else:
        mood = "Neutral 😐"

    length = len(text.split())

    return {
        "sentiment_score": sentiment,
        "mood": mood,
        "length": length
    }
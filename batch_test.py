import urllib.request, json

def predict(text):
    req = urllib.request.Request(
        'http://127.0.0.1:8000/predict',
        data=json.dumps({'text': text}).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    r = json.loads(urllib.request.urlopen(req).read())
    print("  Input    :", r["input_text"])
    print("  Cleaned  :", r["clean_text"])
    print("  Sentiment:", r["sentiment"], f"({r['sentiment_confidence']})")
    print("  Category :", r["category"])
    print()

cases = [
    "it not good speaker",
    "Amazing product quality, loved it",
    "Payment issue, refund not received",
    "Delivery was very late and bad service",
    "Good experience overall",
    "The product quality is bad and delivery was late",
]
for c in cases:
    predict(c)

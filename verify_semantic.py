import urllib.request
import json
import time

def test_api(text):
    url = "http://127.0.0.1:8000/predict"
    data = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            print(f"Input: {res['input_text']:<25} | Category: {res['category']:<10} | Sentiment: {res['sentiment']}")
    except Exception as e:
        print(f"Error testing '{text}': {e}")

if __name__ == "__main__":
    print("Waiting for server to reload...")
    time.sleep(3)
    
    test_cases = [
        "super Ball",
        "the bill is not good",
        "best speaker ever",
        "delivery was fast",
        "i love this phone",
        "refund my money",
        "item was broken"
    ]
    
    print("-" * 65)
    for case in test_cases:
        test_api(case)
    print("-" * 65)

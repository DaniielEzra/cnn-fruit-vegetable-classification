"""
Small test script for the running server.

1. Start the server in one terminal:   uvicorn app:app --reload
2. In a second terminal run:          python test_api.py path/to/image.jpg

It calls GET /health and POST /predict and prints the answers.
"""
import json
import sys

import requests

BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    print("GET /health")
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    print("Status:", response.status_code)
    print(json.dumps(response.json(), indent=2))

    if len(sys.argv) < 2:
        print("\nTo test /predict, pass an image path:  python test_api.py my_banana.jpg")
        return

    image_path = sys.argv[1]
    print(f"\nPOST /predict with {image_path}")
    with open(image_path, "rb") as f:
        response = requests.post(f"{BASE_URL}/predict", files={"file": (image_path, f)}, timeout=60)
    print("Status:", response.status_code)
    data = response.json()
    print(json.dumps(data, indent=2))

    if response.ok:
        print(f"\nPrediction: {data['prediction']}  (best: {data['top_class']} {data['confidence']:.1%})")
        for score in sorted(data["scores"], key=lambda s: -s["probability"]):
            print(f"  {score['class_name']:<9} {score['percent']:6.2f}%  {'#' * int(score['percent'] / 2.5)}")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Cannot connect. Is the server running?  uvicorn app:app --reload")

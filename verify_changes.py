import requests
import os

BASE_URL = "http://localhost:8010"

def test_health():
    print(f"Testing {BASE_URL}/health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("Health Check Passed!")
            print(f"Model Accuracy: {data.get('model_accuracy')}")
            if 'model_accuracy' in data:
                print("SUCCESS: model_accuracy field present.")
            else:
                print("FAILURE: model_accuracy field missing.")
        else:
            print(f"Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

def test_predict():
    print(f"\nTesting {BASE_URL}/api/predict...")
    # Find a sample image
    sample_img = None
    for root, dirs, files in os.walk("raw_data"):
        for file in files:
            if file.endswith(".png"):
                sample_img = os.path.join(root, file)
                break
        if sample_img: break
    
    if not sample_img:
        print("No sample image found in raw_data/ to test prediction.")
        return

    try:
        with open(sample_img, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/api/predict", files=files)
            
        if response.status_code == 200:
            data = response.json()
            print("Prediction Check Passed!")
            if 'predictions' in data and len(data['predictions']) > 0:
                print(f"Top Prediction: {data['predictions'][0]['label']} ({data['predictions'][0]['probability']})")
                print("SUCCESS: Predictions returned.")
            else:
                print("FAILURE: No predictions returned.")
        else:
            print(f"Prediction Failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_health()
    test_predict()

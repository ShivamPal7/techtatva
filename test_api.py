import requests
import os

def test_predict():
    url = "http://localhost:8010/api/predict"
    
    # Try to find a sample image in the project
    sample_img = "raw_data/images/images_001/images/00000001_000.png"
    if not os.path.exists(sample_img):
        # Fallback to any file just to test connection
        print(f"Sample image {sample_img} not found, please provide a valid path.")
        return

    print(f"Testing connectivity to {url}...")
    try:
        with open(sample_img, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success! Prediction results:")
            print(response.json())
        else:
            print(f"Failed: {response.text}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("\nPossible solutions:")
        print("1. Ensure 'python web_app/app.py' is running.")
        print("2. Check if port 8010 is blocked.")

if __name__ == "__main__":
    test_predict()

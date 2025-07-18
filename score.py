import json
import os
import joblib


def init():
    global model
    # The model file is automatically downloaded to this directory
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.joblib")
    model = joblib.load(model_path)
    print("Model loaded successfully.")


def run(data):
    try:
        # Parse input JSON
        input_data = json.loads(data)
        # Assuming the input JSON has a key called "inputs" with a list of features
        predictions = model.predict(input_data["inputs"])
        # Return as JSON
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        error = str(e)
        print(f"Error: {error}")
        return json.dumps({"error": error})

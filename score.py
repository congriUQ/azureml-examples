import json
import os
import pickle


def init():
    global model
    # The model file is automatically downloaded to this directory
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")


def run(data):
    try:
        # Parse input JSON
        input_data = json.loads(data)
        print(f"input_data: {input_data}")
        # Assuming the input JSON has a key called "inputs" with a list of features
        predictions = model.predict(input_data["inputs"])
        # Return as JSON
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        error = str(e)
        print(f"Error: {error}")
        return json.dumps({"error": error})

import json

def load_dataset():
    with open("data/website_data.json", "r") as f:
        dataset = json.load(f)["data"]
    return dataset

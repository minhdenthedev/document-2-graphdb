from src.document2graph.controller import Controller
import os
import json
from tqdm import tqdm

DATA_FOLDER = "/home/m1nhd3n/Works/SideProjects/Document2Graph/data"
OUTPUT_PATH = "/home/m1nhd3n/Works/SideProjects/Document2Graph/outputs/dreeam"
filenames = os.listdir(DATA_FOLDER)

controller = Controller(method="dreeam")
for filename in filenames:
    try:
        with open(os.path.join(DATA_FOLDER, filename), 'r') as f:
            title = filename.replace("_", " ").replace(".txt", "")
            content = f.read()
            output = controller.extract_relationship_dreeam(content, title, max_seq_len=8192)
        for r in output.relationships:
            print(r.evidences)
        with open(os.path.join(OUTPUT_PATH, filename.replace(".txt", "") + ".json"), 'w') as f:
            json.dump(output.model_dump(), f, indent=4)
    except Exception as e:
        print(e)
        print(title)
        print("-" * 15)
        print(content)

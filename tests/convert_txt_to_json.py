import json
import os

FOLDER_PATH = "/home/m1nhd3n/Works/SideProjects/Document2Graph/data"
file_names = os.listdir(FOLDER_PATH)

answers = []

for i, fn in enumerate(file_names):
    with open(os.path.join(FOLDER_PATH, fn), 'r') as f:
        text = f.read()
        answers.append({
            'data': {
                'filename': fn,
                'text': text
            }
        })

with open("/home/m1nhd3n/Works/SideProjects/Document2Graph/data_json/data.json", "w") as f:
    json.dump(answers, f)



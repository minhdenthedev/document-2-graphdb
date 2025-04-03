import json
import os
from src.document2graph.relation_extracting.output_models import Graph

OUTPUT_FOLDER = "/home/m1nhd3n/Works/SideProjects/Document2Graph/parsed_outputs/dreeam"
DATA_FOLDER = "/home/m1nhd3n/Works/SideProjects/Document2Graph/outputs/dreeam"

filenames = os.listdir(DATA_FOLDER)
for filename in filenames:
    with open(os.path.join(DATA_FOLDER, filename), 'r') as f:
        js = json.load(f)
        graph = Graph.model_validate(js)
    out_str = ""
    for relationship in graph.relationships:
        head = [e for e in graph.entities if e.e_id == relationship.head][0]
        tail = [e for e in graph.entities if e.e_id == relationship.tail][0]
        if len([e for e in relationship.evidences]) == 0:
            continue
        out_str += ",".join(
            [head.name, relationship.name, tail.name, "|".join([str(e) for e in relationship.evidences])]) + "\n"

    with open(os.path.join(OUTPUT_FOLDER, filename.replace(".json", ".csv")), "w") as f:
        f.write("head,relationship,tail,evidence\n")
        f.write(out_str[:-1])

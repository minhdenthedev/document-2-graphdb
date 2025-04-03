from typing import Literal

from src.document2graph.relation_extracting.dreeam_extractor import DreeamExtractor
from src.document2graph.relation_extracting.output_models import Entity, Relationship, Graph


class Controller:
    def __init__(self, method: Literal["dreeam", "llm", "both"]):
        match method:
            case "dreeam":
                self.extractor = DreeamExtractor()
            case _:
                pass

    def extract_relationship_dreeam(self, document: str, title: str = None, max_seq_len: int = 1024):
        if not title:
            title = ""
        inputs = self.extractor.parse_sentences_to_inputs(title, document)
        entities = inputs[0]['vertexSet']
        entities_dict = {
            e[0]['name']: Entity(e_id=i,
                                 name=e[0]['name'],
                                 ner_type=e[0]['type'],
                                 properties={},
                                 mention_sent_ids=[m['sent_id'] for m in e])
            for i, e in enumerate(entities)
        }

        rels = self.extractor.extract_relations(inputs, max_seq_len=max_seq_len)
        relationships = [
            Relationship(name=rel['r'],
                         head=entities_dict[rel['head']].e_id,
                         tail=entities_dict[rel['tail']].e_id,
                         evidences=rel['evidence_sent'])
            for rel in rels
        ]
        return Graph(entities=[e for e in entities_dict.values()], relationships=relationships)



import json
from collections import defaultdict

import requests
from IPython.utils.tokenutil import token_at_cursor
from nltk import CoreNLPParser, CoreNLPDependencyParser
from src.document2graph.nlp_parsing.utils import *
import pandas as pd


class NlpParser:
    def __init__(self):
        self.ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
        self.sentence_parser = CoreNLPParser(url='http://localhost:9000')
        self.dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

    def parse_sentences(self, text: str):
        tokens = [t for t in self.sentence_parser.tokenize(text)]
        sentences = []
        token_count = 0
        while token_count < len(tokens):
            sentence_array = []
            while tokens[token_count] != ".":
                sentence_array.append(tokens[token_count])
                token_count += 1
            token_count += 1
            sentence_array.append(".")
            sentences.append(sentence_array)
        return sentences

    def turn_text_into_vertices_set(self, text: str):
        mention_dict = {}
        sents = self.parse_sentences(text)
        for i, sent in enumerate(sents):
            parses = list(self.ner_tagger.tag(sent))

            entities = parse_entity_in_sentence(parses)
            for entity in entities:
                entity['sent_id'] = i

                if entity['name'] in mention_dict:
                    mention_dict[entity['name']].append(entity)
                else:
                    mention_dict[entity['name']] = [entity]

        return [v for v in mention_dict.values()], sents

    def get_coreferences(self, text):
        url = "http://localhost:9000/?properties=" + json.dumps({
            "annotators": "coref",
            "outputFormat": "json"
        })

        response = requests.post(url, data=text.encode("utf-8"))
        ids = []
        texts = []
        sentence_ids = []
        token_starts = []
        token_ends = []
        is_represent = []
        position = []
        coref_type = []

        if response.status_code == 200:
            coref_data = response.json()["corefs"]

            for coref_id, mentions in coref_data.items():
                ids.extend([coref_id for _ in mentions])
                texts.extend([m['text'] for m in mentions])
                sentence_ids.extend([m['sentNum'] - 1 for m in mentions])
                token_starts.extend([m['startIndex'] - 1 for m in mentions])
                token_ends.extend([m['endIndex'] - 1 for m in mentions])
                coref_type.extend([m['type'] for m in mentions])
                is_represent.extend([m['isRepresentativeMention'] for m in mentions])
                position.extend([m['position'] for m in mentions])
            data = {
                "sentence_id": sentence_ids,
                "token_start": token_starts,
                "token_end": token_ends,
                "coref_id": ids,
                "text": texts,
                "is_represent": is_represent,
                "coref_type": coref_type
            }
            df = pd.DataFrame(data)
            return df
        else:
            return "Error: Unable to connect to CoreNLP server."



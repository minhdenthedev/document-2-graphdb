from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer

from src.document2graph.nlp_parsing.nlp_parser import NlpParser
from src.document2graph.relation_extracting.dreeam.docre_model import DocREModel
from src.document2graph.relation_extracting.dreeam.dreeam_dataset import DreeamDataset
from src.document2graph.relation_extracting.dreeam.utils import *


class DreeamExtractor:
    def __init__(self, num_labels: int = 97):
        self.model_path = ("/home/m1nhd3n/Works/SideProjects/Document2Graph/src/document2graph/relation_extracting"
                           "/dreeam/data/bert_student_best.ckpt")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-cased",
        )
        self.config = AutoConfig.from_pretrained(
            "bert-base-cased",
            num_labels=num_labels,
        )
        self.config.transformer_type = "bert"
        self.config.cls_token_id = self.tokenizer.cls_token_id
        self.config.sep_token_id = self.tokenizer.sep_token_id
        bert_model = AutoModel.from_pretrained(
            "bert-base-cased",
            from_tf=False,
            config=self.config,
            attn_implementation="eager"
        )
        state_dict = torch.load(self.model_path, map_location="cuda")
        state_dict.pop("model.embeddings.position_ids", None)
        self.model = DocREModel(self.config, bert_model, self.tokenizer,
                                num_labels=4, max_sent_num=25)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to("cuda")
        self.nlp_parser = NlpParser()

    def parse_sentences_to_inputs(self, title: str, text: str):
        vertex_set, ses = self.nlp_parser.turn_text_into_vertices_set(text)
        return [{
            'title': title,
            'sents': ses,
            'vertexSet': vertex_set
        }]

    def get_features(self, data: list, max_seq_length=1024):
        features = []

        for doc_id in range(len(data)):

            sample = data[doc_id]
            entities = sample['vertexSet']
            entity_start, entity_end = [], []
            # record entities
            for entity in entities:
                for mention in entity:
                    sent_id = mention["sent_id"]
                    pos = mention["pos"]
                    entity_start.append((sent_id, pos[0],))
                    entity_end.append((sent_id, pos[1] - 1,))

            # add entity markers
            sents, sent_map, sent_pos = self.add_entity_markers(sample, entity_start, entity_end)

            # entity start, end position
            entity_pos = []

            for e in entities:
                entity_pos.append([])
                assert len(e) != 0
                for m in e:
                    start = sent_map[m["sent_id"]][m["pos"][0]]
                    end = sent_map[m["sent_id"]][m["pos"][1]]
                    entity_pos[-1].append((start, end,))

            relations, hts, sent_labels = [], [], []

            for h in range(len(entities)):
                for t in range(len(entities)):
                    # all entity pairs that do not have relation are treated as negative samples
                    if h != t and [h, t] not in hts:  # and [t, h] not in hts:
                        relation = [1] + [0] * (len(rel2id) - 1)
                        sent_evi = [0] * len(sent_pos)
                        relations.append(relation)

                        hts.append([h, t])
                        sent_labels.append(sent_evi)

            assert len(relations) == len(entities) * (len(entities) - 1)
            assert len(sents) < max_seq_length
            sents = sents[:max_seq_length - 2]
            input_ids = self.tokenizer.convert_tokens_to_ids(sents)
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

            feature = [{'input_ids': input_ids,
                        'entity_pos': entity_pos,
                        'labels': relations,
                        'hts': hts,
                        'sent_pos': sent_pos,
                        'sent_labels': sent_labels,
                        'title': sample['title'],
                        }]

            features.extend(feature)

        return features

    def extract_relations(self, data: list, max_seq_len: int = 1024):
        features = self.get_features(data, max_seq_len)
        dataset = DreeamDataset(features)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                drop_last=False)
        preds, evi_preds = [], []
        scores, topks = [], []
        attns = []

        for batch in dataloader:
            self.model.eval()

            tag = "infer"

            inputs = load_input(batch, "cuda", tag)

            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = outputs["rel_pred"]
                pred = pred.cpu().numpy()
                pred[np.isnan(pred)] = 0
                preds.append(pred)

                if "scores" in outputs:
                    scores.append(outputs["scores"].cpu().numpy())
                    topks.append(outputs["topks"].cpu().numpy())

                if "evi_pred" in outputs:  # relation extraction and evidence extraction
                    evi_pred = outputs["evi_pred"]
                    evi_pred = evi_pred.cpu().numpy()
                    evi_preds.append(evi_pred)

                if "attns" in outputs:  # attention recorded
                    attn = outputs["attns"]
                    attns.extend([a.cpu().numpy() for a in attn])

        preds = np.concatenate(preds, axis=0)
        if len(scores) > 0:
            scores = np.concatenate(scores, axis=0)
            topks = np.concatenate(topks, axis=0)

        if len(evi_preds):
            evi_preds = np.concatenate(evi_preds, axis=0)

        scores = np.array(scores)
        topks = np.array(topks)
        evi_preds = np.array(evi_preds)
        official_results, results = to_official(preds, features, evi_preds=evi_preds, scores=scores, topks=topks)
        return parse_official_results(official_results, data)

    def add_entity_markers(self, sample, entity_start, entity_end):
        """ add entity marker (*) at the end and beginning of entities. """

        sents = []
        sent_map = []
        sent_pos = []
        i_ts = 0
        sent_start = 0
        for i_s, sent in enumerate(sample['sents']):
            # add * marks to the beginning and end of entities
            new_map = {}

            for i_t, token in enumerate(sent):
                tokens_wordpiece = self.tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
                if i_t == len(sent) - 1:
                    i_ts = i_t

            sent_end = len(sents)
            # [sent_start, sent_end)
            sent_pos.append((sent_start, sent_end,))
            sent_start = sent_end

            # update the start/end position of each token.
            new_map[i_ts + 1] = len(sents)
            sent_map.append(new_map)

        return sents, sent_map, sent_pos


data_in = {
    'title': 'About Hoang Minh',
    'sents': [
        "Hoang Minh was born in Hanoi .".split(),
        "It is the capital of Vietnam .".split()
    ],
    'vertexSet': [
        [
            {
                'name': 'Hoang Minh',
                'sent_id': 0,
                'pos': [0, 2],
                'type': 'PER'
            }
        ],
        [
            {
                'name': 'Hanoi',
                'sent_id': 0,
                'pos': [5, 6],
                'type': 'LOC'
            }
        ],
        [
            {
                'name': 'Vietnam',
                'sent_id': 1,
                'pos': [5, 6],
                'type': 'LOC'
            }
        ]
    ]
}

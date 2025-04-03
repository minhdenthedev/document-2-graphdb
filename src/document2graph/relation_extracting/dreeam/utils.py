import json

import numpy as np
import torch

with open('/home/m1nhd3n/Works/SideProjects/Document2Graph/src/document2graph/relation_extracting'
          '/dreeam/data/rel2id.json', 'r') as f:
    rel2id = json.load(f)
id2rel = {value: key for key, value in rel2id.items()}
ent2id = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}
with open('/home/m1nhd3n/Works/SideProjects/Document2Graph/src/document2graph/relation_extracting'
          '/dreeam/data/rel_info.json', "r") as f:
    rel_info = json.load(f)


def load_input(batch, device, tag="dev"):
    inputs = {'input_ids': batch[0].to(device),
              'attention_mask': batch[1].to(device),
              'labels': batch[2].to(device),
              'entity_pos': batch[3],
              'hts': batch[4],
              'sent_pos': batch[5],
              'sent_labels': batch[6].to(device) if (not batch[6] is None) and (batch[7] is None) else None,
              'teacher_attns': batch[7].to(device) if not batch[7] is None else None,
              'tag': tag
              }

    return inputs


def collate_fn(batch):
    max_len = max([len(b["input_ids"]) for b in batch])
    max_sent = max([len(b["sent_pos"]) for b in batch])
    input_ids = [b["input_ids"] + [0] * (max_len - len(b["input_ids"])) for b in batch]
    input_mask = [[1.0] * len(b["input_ids"]) + [0.0] * (max_len - len(b["input_ids"])) for b in batch]
    labels = [b["labels"] for b in batch]
    entity_pos = [b["entity_pos"] for b in batch]
    hts = [b["hts"] for b in batch]
    sent_pos = [b["sent_pos"] for b in batch]
    sent_labels = [b["sent_labels"] for b in batch if "sent_labels" in b]
    attns = [b["attns"] for b in batch if "attns" in b]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    labels = [torch.tensor(label) for label in labels]
    labels = torch.cat(labels, dim=0)

    if sent_labels != [] and None not in sent_labels:
        sent_labels_tensor = []
        for sent_label in sent_labels:
            sent_label = np.array(sent_label)
            sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent - sent_label.shape[1]))))
        sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
    else:
        sent_labels_tensor = None

    if len(attns) > 0:

        attns = [np.pad(attn, ((0, 0), (0, max_len - attn.shape[1]))) for attn in attns]
        attns = torch.from_numpy(np.concatenate(attns, axis=0))
    else:
        attns = None

    output = (input_ids, input_mask, labels, entity_pos, hts, sent_pos, sent_labels_tensor, attns)

    return output


def to_official(preds: np.ndarray, features: list, evi_preds: np.ndarray, scores: np.ndarray, topks: np.ndarray):
    """
    Convert the predictions to official format for evaluating.
    Input:
        preds: list of dictionaries, each dictionary entry is a predicted relation triple from the original document.
                Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        features: list of features within each document. Identical to the lists obtained from pre-processing.
        evi_preds: list of the evidence prediction corresponding to each relation triple prediction.
        scores: list of scores of topk relation labels for each entity pair.
        topks: list of topk relation labels for each entity pair.
    Output:
        official_res: official results used for evaluation.
        res: topk results to be dumped into file, which can be further used during fushion.
    """

    h_idx, t_idx, title, sents = [], [], [], []

    for feat in features:
        if "entity_map" in feat:
            hts = [[feat["entity_map"][ht[0]], feat["entity_map"][ht[1]]] for ht in feat["hts"]]
        else:
            hts = feat["hts"]

        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [feat["title"] for _ in hts]
        sents += [len(feat["sent_pos"])] * len(hts)

    official_res = []
    res = []
    score = None
    for i in range(preds.shape[0]):  # for each entity pair
        if scores.shape[0] > 0:
            score = extract_relative_score(scores[i], topks[i])
            pred = topks[i]
        else:
            pred = preds[i]
            pred = np.nonzero(pred)[0].tolist()

        for p in pred:  # for each predicted relation label (topk)
            curr_result = {
                'title': title[i],
                'h_idx': h_idx[i],
                't_idx': t_idx[i],
                'r': id2rel[p],
            }

            if evi_preds.shape[0] > 0:
                curr_evi = evi_preds[i]
                evis = np.nonzero(curr_evi)[0].tolist()
                curr_result["evidence"] = [evi for evi in evis if evi < sents[i]]
            if scores.shape[0] > 0:
                curr_result["score"] = score[np.where(topks[i] == p)].item()
            if p != 0 and p in np.nonzero(preds[i])[0].tolist():
                official_res.append(curr_result)
            res.append(curr_result)

    return official_res, res


def extract_relative_score(scores: np.ndarray, topks: np.ndarray) -> np.ndarray:
    """
    Get relative score from top k predictions.
    Input:
        :scores: a list containing scores of top k predictions.
        :topks: a list containing relation labels of top k predictions.
    Output:
        :scores: a list containing relative scores of top k predictions.
    """

    na_score = scores[-1].item() - 1
    if 0 in topks:
        na_score = scores[np.where(topks == 0)].item()

    scores -= na_score

    return scores


def parse_official_results(results, data):
    answers = []
    for result in results:
        answer = {
            'head': data[0]['vertexSet'][result['h_idx']][0]['name'],
            'r': rel_info[result['r']],
            'tail': data[0]['vertexSet'][result['t_idx']][0]['name'],
            'evidence_sent': result['evidence']
        }
        answers.append(answer)
    return answers

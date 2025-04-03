docred_ent2id = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}
LOC_NAMES = ['LOCATION', 'STATE_OR_PROVINCE']
ORG_NAMES = ['ORGANIZATION']
PER_NAMES = ['PERSON']
TIME_NAMES = ['DATE', 'TIME']
NUM_NAMES = ['NUMBER']
MISC_NAMES = ['MISC']


def stanford_ner_to_docred(ner_tag: str):
    if ner_tag in ORG_NAMES:
        return 'ORG'
    elif ner_tag in LOC_NAMES:
        return 'LOC'
    elif ner_tag in NUM_NAMES:
        return 'NUM'
    elif ner_tag in TIME_NAMES:
        return 'TIME'
    elif ner_tag in MISC_NAMES:
        return 'MISC'
    elif ner_tag in PER_NAMES:
        return 'PER'
    else:
        return 'NA'


def parse_entity_in_sentence(nlp_parsed: list[tuple]):
    answer = []
    last_ner = None
    entities_count = 0
    for i, tup in enumerate(nlp_parsed):
        ner = tup[1]
        if ner != 'O':
            if last_ner is None or len(answer) == 0:
                entity = {'name': tup[0],
                          'type': stanford_ner_to_docred(ner),
                          'pos': [i, i + 1]}
                answer.append(entity)
                last_ner = ner
            else:
                if ner == last_ner:
                    answer[entities_count]['name'] += " " + tup[0]
                    answer[entities_count]['pos'][1] += 1
                else:
                    entities_count += 1
                    entity = {'name': tup[0],
                              'type': stanford_ner_to_docred(ner),
                              'pos': [i, i + 1]}
                    answer.append(entity)
                    last_ner = ner
        else:
            last_ner = ner
    return answer

import json
from collections import Counter
from typing import Dict

from pydantic import BaseModel

from src.document2graph.relation_extracting.output_models import Graph, GeminiMergedEntityOutput, GeminiEntityInput, \
    Entity, Relationship, GeminiExtendTriplet
from google import genai
from src.document2graph.relation_extracting.dreeam.utils import rel_info, ent2id



class GraphProcessor:
    def __init__(self,
                 text: str,
                 remove_isolated=True,
                 remove_no_evidence_relationship=True):
        self.text = text
        self.remove_isolated = remove_isolated
        self.remove_no_evidence_relationship = remove_no_evidence_relationship
        self.llm = genai.Client(
            api_key="AIzaSyC0SFy8iUZycmWXbUtTA6IEgof2O1PS5jc"
        )

    def remove_isolated_nodes(self, graph: Graph):
        connected_entities = set()
        for relationship in graph.relationships:
            connected_entities.add(relationship.head)
            connected_entities.add(relationship.tail)
        isolated_entities = [e for e in graph.entities if e.e_id not in connected_entities]
        graph.entities = [e for e in graph.entities if e.e_id in connected_entities]
        return graph, isolated_entities

    def remove_no_evi_relation(self, graph: Graph):
        graph.relationships = [r for r in graph.relationships if len(r.evidences) > 0]
        return graph

    def merge_same_entities(self, graph: Graph):
        entities_llm_input = [
            GeminiEntityInput(entity_id=e.e_id, name=e.name)
            for e in graph.entities
        ]
        prompt = f"""Based on the text below and your own knowledge, try to merge the entities that refer to the same thing. 
        Make sure that the official_name is as coherent as possible.
        Only return the entities that are merged from 2 or more entities.
        Text:
        {self.text}
        
        Entities list:
        {entities_llm_input}
        """
        response = self.llm.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[GeminiMergedEntityOutput],
            },
        )

        merge_entities: list[GeminiMergedEntityOutput] = response.parsed

        # Substitute the e_id
        for me in merge_entities:
            related_ids = me.merged_from_entities_with_ids

            for e in graph.entities:
                if e.e_id in related_ids:
                    e.e_id = related_ids[0]
                    e.name = me.official_name

            for r in graph.relationships:
                if r.head in related_ids:
                    r.head = related_ids[0]
                if r.tail in related_ids:
                    r.tail = related_ids[0]

        # Merge duplicated entities
        entities_dict: Dict[str, Entity] = {}
        for e in graph.entities:
            if e.e_id in entities_dict:
                entities_dict[str(e.e_id)].properties.update(e.properties)
                entities_dict[str(e.e_id)].mention_sent_ids.extend(e.mention_sent_ids)
            else:
                entities_dict[str(e.e_id)] = e
        for v in entities_dict.values():
            v.mention_sent_ids = list(set(v.mention_sent_ids))

        # Merge duplicated relationships
        relation_dicts: Dict[str, Relationship] = {}
        for r in graph.relationships:
            r_key = str(r.head) + "_" + str(r.name) + "_" + str(r.tail)
            if r_key in relation_dicts:
                relation_dicts[r_key].evidences.extend(r.evidences)
            else:
                relation_dicts[r_key] = r
        for r in relation_dicts.values():
            r.evidences = list(set(r.evidences))

        graph.entities = [e for e in entities_dict.values()]
        graph.relationships = [r for r in relation_dicts.values()]
        return graph

    def llm_validate(self, graph: Graph):
        entities_dict = {e.e_id: e for e in graph.entities}
        relations_input = {i: (entities_dict[r.head].name, r.name, entities_dict[r.tail].name)
                           for i, r in enumerate(graph.relationships)}
        real_relations_dict = {i: r for i, r in enumerate(graph.relationships)}

        prompt = f"""
                {self.text}
                
                Based on the text above, return to me the list of triplets' IDs that cannot be inferred from.
                Triplets list:
                {relations_input}
                """
        response = self.llm.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[int],
            },
        )

        removed_ids: list[int] = response.parsed
        valid_relations = [v for k, v in real_relations_dict.items() if k not in removed_ids]
        graph.relationships = valid_relations
        graph, _ = self.remove_isolated_nodes(graph)
        return graph

    def llm_extend_graph(self, graph: Graph, isolated_entities: Entity):
        entities_dict = {e.e_id: e for e in graph.entities}
        relations_input = {i: (entities_dict[r.head].name, r.name, entities_dict[r.tail].name)
                           for i, r in enumerate(graph.relationships)}
        real_relations_dict = {i: r for i, r in enumerate(graph.relationships)}
        relation_list = [r for r in rel_info.values()]
        ner_list = [n for n in ent2id.keys()]
        prompt = f"""
                        {self.text}
                        
                        Examples triplets:
                        {relations_input}

                        Based on the text and the example triplets above, try to find more triplets that express the relationship
                        between entities. Remember to add the indices of sentences that you used as evidences for you answer.
                        If you can't find any triplets, return empty list. Do not return any triplets that are included in
                        the examples.
                        
                        The entities must be explicitly mentioned in the text.
                        The head_type and tail_type must be one of these types: {ner_list}
                        The relationships must be one of these types:
                        {relation_list}
                        """
        response = self.llm.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[GeminiExtendTriplet],
            },
        )

        extended_triplets: list[GeminiExtendTriplet] = response.parsed
        if len(extended_triplets) == 0:
            print("No more triplets found.")
            return graph
        max_e_id = 0
        for e in graph.entities:
            if e.e_id > max_e_id:
                max_e_id = e.e_id

        added_names = {}
        for e in extended_triplets:
            if e.head not in added_names:
                max_e_id += 1
                graph.entities.append(
                    Entity(e_id=max_e_id,
                           name=e.head,
                           ner_type=e.head_type,
                           mention_sent_ids=[],
                           properties={})
                )
                added_names[e.head] = max_e_id
            if e.tail not in added_names:
                max_e_id += 1
                graph.entities.append(
                    Entity(e_id=max_e_id,
                           name=e.tail,
                           ner_type=e.tail_type,
                           mention_sent_ids=[],
                           properties={})
                )
                added_names[e.tail] = max_e_id
            print(e)


    def process(self, graph):
        print(graph.model_dump_json(indent=2))
        if self.remove_no_evidence_relationship:
            graph = self.remove_no_evi_relation(graph)
        graph, isolated_entities = self.remove_isolated_nodes(graph)
        graph = self.merge_same_entities(graph)
        graph = self.llm_validate(graph)
        return graph
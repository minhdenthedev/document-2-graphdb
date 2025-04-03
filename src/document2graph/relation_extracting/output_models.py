from pydantic import BaseModel


class Entity(BaseModel):
    e_id: int
    name: str
    ner_type: str
    properties: dict[str, str]
    mention_sent_ids: list[int]


class Relationship(BaseModel):
    name: str
    head: int
    tail: int
    evidences: list[int]


class Graph(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]

class GeminiEntityInput(BaseModel):
    entity_id: int
    name: str

class GeminiMergedEntityOutput(BaseModel):
    official_name: str
    merged_from_entities_with_ids: list[int]

class GeminiExtendTriplet(BaseModel):
    head: str
    head_type: str
    relationship: str
    tail: str
    tail_type: str
    evidence: list[int]

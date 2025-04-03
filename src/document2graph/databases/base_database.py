from abc import ABC, abstractmethod
from typing import List

from src.document2graph.relation_extracting.output_models import Entity, Relationship, Graph


class BaseDatabase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def insert_entities(self, entities: List[Entity]):
        pass


    @abstractmethod
    def insert_relationships(self, relationships: List[Relationship]):
        pass


    @abstractmethod
    def insert_triplets(self):
        pass
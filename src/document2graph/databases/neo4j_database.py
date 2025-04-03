from typing import List

from neo4j import GraphDatabase

from src.document2graph.databases.base_database import BaseDatabase
from src.document2graph.relation_extracting.output_models import Relationship, Entity, Graph


class Neo4jDatabase(BaseDatabase):
    def __init__(self, uri: str = "bolt://localhost:7689",
                 username: str = "neo4j",
                 password: str = "minhminh"):
        super().__init__()
        self.auth = (username, password)
        self.uri = uri

    def insert_entities(self, entities: List[Entity]):
        try:
            with GraphDatabase.driver(self.uri, auth=self.auth) as driver:
                try:
                    for entity in entities:
                        driver.execute_query(
                            f"MERGE (e:{entity.ner_type} {{e_id: $e_id, name: $name}})",
                            e_id=entity.e_id, name=entity.name,
                            database_="neo4j"
                        )
                except Exception as e:
                    raise e

        except Exception as e:
            print(e)

    def insert_relationships(self, relationships: List[Relationship]):
        try:
            with GraphDatabase.driver(self.uri, auth=self.auth) as driver:
                try:
                    for relationship in relationships:
                        driver.execute_query(f"""
                        MATCH (h {{e_id: $head}})
                        MATCH (t {{e_id: $tail}})
                        MERGE (h)-[r:{relationship.name.upper().replace(" ", "_")}]->(t)
                        SET r += $properties
                        """, head=relationship.head,
                        tail=relationship.tail,
                        name=relationship.name,
                        properties={"evidence": relationship.evidences},
                        database_="neo4j")

                except Exception as e:
                    raise e
        except Exception as e:
            raise e


    def insert_triplets(self):
        pass

    def insert_graph(self, graph: Graph):
        self.insert_entities(graph.entities)
        self.insert_relationships(graph.relationships)

    def reset_db(self):
        try:
            with GraphDatabase.driver(self.uri, auth=self.auth) as driver:
                try:
                    driver.execute_query("MATCH (n) DETACH DELETE n",
                                         database_="neo4j")
                except Exception as e:
                    raise e
        except Exception as e:
            raise e
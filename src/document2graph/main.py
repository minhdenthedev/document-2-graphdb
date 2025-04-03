from src.document2graph.databases.neo4j_database import Neo4jDatabase
from src.document2graph.controller import Controller
from src.document2graph.relation_extracting.graph_processor import GraphProcessor

if __name__ == '__main__':
    with open(
            "/home/m1nhd3n/Works/SideProjects/Document2Graph/data/New_Hampshire_Sen._Jeanne_Shaheen_won’t_seek_reelection_in_2026.txt",
            "r") as f:
        text = f.read()
        title = "New_Hampshire_Sen._Jeanne_Shaheen_won’t_seek_reelection_in_2026".replace("_", " ")

    controller = Controller("dreeam")
    graph = controller.extract_relationship_dreeam(text, title, max_seq_len=9000)
    graph = GraphProcessor(text).process(graph)
    db = Neo4jDatabase()
    db.reset_db()
    db.insert_graph(graph)
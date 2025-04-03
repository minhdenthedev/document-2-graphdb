import json
import unittest

from src.document2graph.nlp_parsing.nlp_parser import NlpParser


class MyTestCase(unittest.TestCase):
    def test_turn_text_2_vertices(self):
        with open(
                "/home/m1nhd3n/Works/SideProjects/Document2Graph/data/New_Hampshire_Sen._Jeanne_Shaheen_won’t_seek_reelection_in_2026.txt",
                "r") as f:
            text = f.read()
            title = "New_Hampshire_Sen._Jeanne_Shaheen_won’t_seek_reelection_in_2026".replace("_", " ")

        parser = NlpParser()
        vertices, sents = parser.turn_text_into_vertices_set(text)
        vertices = [v for v in vertices if len(vertices) > 1]
        print(json.dumps(vertices, indent=2))

if __name__ == '__main__':
    unittest.main()

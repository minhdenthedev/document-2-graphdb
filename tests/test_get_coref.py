import unittest

import pandas as pd

from src.document2graph.nlp_parsing.nlp_parser import NlpParser


class MyTestCase(unittest.TestCase):
    def test_get_coref(self):
        with open(
                "/home/m1nhd3n/Works/SideProjects/Document2Graph/data/New_Hampshire_Sen._Jeanne_Shaheen_won’t_seek_reelection_in_2026.txt",
                "r") as f:
            text = f.read()
            title = "New_Hampshire_Sen._Jeanne_Shaheen_won’t_seek_reelection_in_2026".replace("_", " ")

        parser = NlpParser()
        corefs = parser.get_coreferences(text)





if __name__ == '__main__':
    unittest.main()

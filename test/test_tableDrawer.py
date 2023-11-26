from unittest import TestCase

from algorithmX import AlgorithmX
from blocks import Box, BasicBlockTypes
from tableDrawer import TableDrawer


class TestTableDrawer(TestCase):

    def test_basic_result_drawing(self):
        box = Box("TestBox", (2, 2))
        blocks = [BasicBlockTypes.blocks["-"], BasicBlockTypes.blocks["C"]]
        position_table, table = AlgorithmX.generate_table(box, blocks)
        selected_cover = AlgorithmX.get_cover(table)
        TableDrawer.draw_all_results(selected_cover, position_table, box)
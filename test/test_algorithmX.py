from unittest import TestCase

import numpy as np

from algorithmX import AlgorithmX
from blocks import BasicBlockTypes, Box
from tableDrawer import TableDrawer


class TestAlgorithm(TestCase):

    def setUp(self):
        self.matrix = np.array(((1, 0, 1, 0, 0, 0),
                                (1, 0, 0, 1, 0, 0),
                                (1, 0, 0, 0, 1, 0),
                                (1, 0, 0, 0, 0, 1),
                                (0, 1, 1, 1, 1, 0),
                                (0, 1, 1, 1, 0, 1),
                                (0, 1, 0, 1, 1, 1),
                                (0, 1, 1, 0, 1, 1)))

        self.matrix2 = np.array(((1, 0, 0, 1, 0, 0, 1),
                                 (1, 0, 0, 1, 0, 0, 0),
                                 (0, 0, 0, 1, 1, 0, 1),
                                 (0, 0, 1, 0, 1, 1, 0),
                                 (0, 1, 1, 0, 0, 1, 1),
                                 (0, 1, 0, 0, 0, 0, 1),
                                 ))

    def test_find_minimum_row(self):
        self.assertEqual(AlgorithmX.get_column_index_lowest_ones(self.matrix), (0, 4))
        self.assertEqual(AlgorithmX.get_column_index_lowest_ones(self.matrix2), (0, 2))

    def test_find_rows_with_one_in_index(self):
        selected = AlgorithmX.get_all_rows_index_with_one_in_index(self.matrix, 0)
        self.assertEqual(len(selected), 4)
        self.assertTrue(np.all(selected == (0, 1, 2, 3)))

        selected = AlgorithmX.get_all_rows_index_with_one_in_index(self.matrix2, 0)
        self.assertEqual(len(selected), 2)
        self.assertTrue(np.all(selected == (0, 1)))

    def test_get_rows_index_with_lowest_sum_colum(self):
        selected = AlgorithmX.get_rows_index_with_lowest_sum_colum_test(self.matrix)
        self.assertEqual(len(selected), 4)
        self.assertTrue(np.all(selected == (0, 1, 2, 3)))

        selected = AlgorithmX.get_rows_index_with_lowest_sum_colum_test(self.matrix2)
        self.assertEqual(len(selected), 2)
        self.assertTrue(np.all(selected == (0, 1)))

    def test_get_row_with_zero(self):
        selected = np.where(AlgorithmX.get_all_rows_with_zeros_after_and(self.matrix, self.matrix[0, :]))[0]
        self.assertEqual(len(selected), 1)
        self.assertTrue(np.all(selected == (6,)))

        selected = AlgorithmX.get_all_row_with_zeros_after_and_using_index_test(self.matrix, 0)
        self.assertEqual(len(selected), 1)
        self.assertTrue(np.all(selected == np.array((1, 1, 1, 1))))

        selected = np.where(AlgorithmX.get_all_rows_with_zeros_after_and(self.matrix2, self.matrix2[0, :]))[0]
        self.assertEqual(len(selected), 1)
        self.assertTrue(np.all(selected == (3,)))

        selected = AlgorithmX.get_all_row_with_zeros_after_and_using_index_test(self.matrix2, 0)
        self.assertEqual(len(selected), 1)
        self.assertTrue(np.all(selected == np.array((0, 1, 1, 1))))

        selected = np.where(AlgorithmX.get_all_rows_with_zeros_after_and(self.matrix2, self.matrix2[1, :]))[0]
        self.assertEqual(len(selected), 3)
        self.assertTrue(np.all(selected == (3, 4, 5)))

        selected = AlgorithmX.get_all_row_with_zeros_after_and_using_index_test(self.matrix2, 1)
        self.assertEqual(len(selected), 3)
        self.assertTrue(np.all(selected == np.array(((0, 1, 1, 1, 0),
                                                     (1, 1, 0, 1, 1),
                                                     (1, 0, 0, 0, 1),
                                                     ))
                               ))

    def test_cover_matrix(self):
        selected_cover = AlgorithmX.get_cover(self.matrix, 2)
        self.assertEqual(len(selected_cover), 4)
        # unpacked_solutions = AlgorithmX.unpack_solutions(selected_cover)
        print(selected_cover)
        self.assertTrue(selected_cover == [[0, 6], [1, 7], [2, 5], [3, 4]])

    def test_create_table(self):
        box = Box("TestBox", (2, 2))
        blocks = [BasicBlockTypes.blocks["-"], BasicBlockTypes.blocks["C"]]
        position_table, table = AlgorithmX.generate_table(box, blocks)
        self.assertTrue(np.all(table == self.matrix))

    def test_complete(self):
        box = Box("TestBox", (2, 2))
        blocks = [BasicBlockTypes.blocks["-"], BasicBlockTypes.blocks["C"]]
        position_table, table = AlgorithmX.generate_table(box, blocks)
        selected_cover = AlgorithmX.get_cover(table, 2)
        self.assertEqual(len(selected_cover), 4)
        self.assertTrue(selected_cover == [[0, 6], [1, 7], [2, 5], [3, 4]])
        print(f"Done in {AlgorithmX._cover_counts} cover counts")

    def test_unpack(self):
        box = Box("TestBox", (2, 2))
        blocks = [BasicBlockTypes.blocks["-"], BasicBlockTypes.blocks["+"], BasicBlockTypes.blocks["D"]]
        position_table, table = AlgorithmX.generate_table(box, blocks)
        selected_cover = AlgorithmX.get_cover(table, 3)
        expected_result = [[0, 5, 9], [0, 6, 11], [1, 4, 9], [1, 7, 10], [2, 4, 11], [2, 7, 8], [3, 5, 10], [3, 6, 8]]
        print(selected_cover)
        self.assertEqual(len(selected_cover), 8)
        self.assertTrue(selected_cover == expected_result)
        print(f"Done in {AlgorithmX._cover_counts} cover counts")

    def test_no_cover(self):
        blocks = [BasicBlockTypes.blocks["C"], BasicBlockTypes.blocks["C"]]
        box = Box("Test", (2, 2))

        position_table, table = AlgorithmX.generate_table(box, blocks)
        selected_cover = AlgorithmX.get_cover(table, 2)
        self.assertIsNone(selected_cover)
        print(f"Done in {AlgorithmX._cover_counts} cover counts")

    def test_no_cover2(self):
        blocks = [BasicBlockTypes.blocks["+"]]
        box = Box("Test", (2, 2))

        position_table, table = AlgorithmX.generate_table(box, blocks)
        selected_cover = AlgorithmX.get_cover(table, 1)
        self.assertIsNone(selected_cover)
        print(f"Done in {AlgorithmX._cover_counts} cover counts")

    def test_duplicated(self):
        blocks = [BasicBlockTypes.blocks["+"], BasicBlockTypes.blocks["-"], BasicBlockTypes.blocks["C"]]
        box = Box("Test", (2, 2))

        position_table, table = AlgorithmX.generate_table(box, blocks)
        selected_cover = AlgorithmX.get_cover(table, 3)
        print(selected_cover)
        self.assertEqual(8, len(selected_cover))
        self.assertTrue(
            selected_cover == [[0, 10], [1, 11], [2, 9], [3, 8], [4, 10], [5, 11], [6, 9], [7, 8]])
        print(f"Done in {AlgorithmX._cover_counts} cover counts")

    def test_cover_duplicated(self):
        blocks = [BasicBlockTypes.blocks["+"], BasicBlockTypes.blocks["-"], BasicBlockTypes.blocks["D"]]
        box = Box("Test", (2, 2))

        position_table, table = AlgorithmX.generate_table(box, blocks)
        selected_cover = AlgorithmX.get_cover(table, 3)
        print(selected_cover)
        self.assertEqual(8, len(selected_cover))
        self.assertTrue(
            selected_cover == [[0, 5, 9], [0, 6, 11], [1, 4, 9], [1, 7, 10], [2, 4, 11], [2, 7, 8], [3, 5, 10],
                               [3, 6, 8]])
        print(f"Done in {AlgorithmX._cover_counts} cover counts")

    def test_cover_duplicated_2(self):
        blocks = [BasicBlockTypes.blocks["D"], BasicBlockTypes.blocks["+"], BasicBlockTypes.blocks["-"]]
        box = Box("Test", (2, 2))

        position_table, table = AlgorithmX.generate_table(box, blocks)
        selected_cover = AlgorithmX.get_cover(table, 3)
        print(selected_cover)
        self.assertEqual(8, len(selected_cover))
        self.assertTrue(
            selected_cover == [[0, 6, 11], [0, 7, 10], [1, 4, 9], [1, 5, 8], [2, 5, 11], [2, 7, 9], [3, 4, 10],
                               [3, 6, 8]])
        print(f"Done in {AlgorithmX._cover_counts} cover counts")

    def test_long_duplicated(self):
        blocks = [BasicBlockTypes.blocks["C"], BasicBlockTypes.blocks["C"], BasicBlockTypes.blocks["D"],
                  BasicBlockTypes.blocks["+"]]

        box = Box("Test", (2, 3))

        position_table, table = AlgorithmX.generate_table(box, blocks)
        selected_cover = AlgorithmX.get_cover(table, 4)
        print(selected_cover)
        TableDrawer.draw_all_results(selected_cover, position_table, box, True)
        self.assertEqual(24, len(selected_cover))
        print(f"Done in {AlgorithmX._cover_counts} cover counts")

    def test_cache_usage(self):
        blocks = [BasicBlockTypes.blocks["I"], BasicBlockTypes.blocks["J"], BasicBlockTypes.blocks["D"],
                  BasicBlockTypes.blocks["+"], BasicBlockTypes.blocks["-"]]

        box = Box("Test", (3, 4))

        position_table, table = AlgorithmX.generate_table(box, blocks)
        selected_cover = AlgorithmX.get_cover(table, 5)
        print(selected_cover)
        TableDrawer.draw_all_results(selected_cover, position_table, box, True)
        print(f"Done in {AlgorithmX._cover_counts} cover counts")
        self.assertEqual(40, len(selected_cover))

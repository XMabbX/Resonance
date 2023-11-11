from unittest import TestCase

import numpy as np

from algorithmX import AlgorithmX


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
        selected = AlgorithmX.get_all_columns_index_with_one_in_index(self.matrix, 0)
        self.assertEqual(len(selected), 4)
        self.assertTrue(np.all(selected == (0, 1, 2, 3)))

        selected = AlgorithmX.get_all_columns_index_with_one_in_index(self.matrix2, 0)
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
        selected_cover = AlgorithmX.get_cover(self.matrix)
        self.assertEqual(len(selected_cover), 4)
        unpacked_solutions = AlgorithmX.unpack_solutions(selected_cover)
        print(unpacked_solutions)
        self.assertTrue(unpacked_solutions == [(0, 6), (1, 7), (2, 5), (3, 4)])

    def test_cover_matrix2(self):
        selected_cover = AlgorithmX.get_cover(self.matrix2)
        self.assertEqual(len(selected_cover), 2)
        unpacked_solutions = AlgorithmX.unpack_solutions(selected_cover)
        print(unpacked_solutions)
        self.assertEqual(len(unpacked_solutions), 1)
        self.assertTrue(unpacked_solutions == [(1, 3, 5)])

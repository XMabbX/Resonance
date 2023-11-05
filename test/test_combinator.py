import unittest

import numpy as np

from blocks import Box, BasicBlockTypes, Orientation
from combinator import Combinator


class TestCombinator(unittest.TestCase):

    def test_generate_coordinates(self):
        box = Box("TestBox", (2, 2))
        assert list(Combinator.generate_position(box))
        expected_coordinates = [np.array((0, 0)), np.array((0, 1)), np.array((1, 0)), np.array((1, 1))]
        generated_coordinates = list(Combinator.generate_position(box))
        assert len(expected_coordinates) == len(generated_coordinates)
        for exp_coord, gen_coord in zip(expected_coordinates, generated_coordinates):
            assert np.all(exp_coord == gen_coord)

    def test_generate_valid_positions_1x1(self):
        box = Box("TestBox", (2, 2))
        block = BasicBlockTypes.blocks["-"]

        assert list(Combinator.generate_valid_positions(block, box))

        expected_coordinates = [np.array((0, 0)), np.array((0, 1)), np.array((1, 0)), np.array((1, 1))]
        generated_coordinates = list(Combinator.generate_position(box))

        assert len(expected_coordinates) == len(generated_coordinates)
        for exp_coord, gen_coord in zip(expected_coordinates, generated_coordinates):
            assert np.all(exp_coord == gen_coord)

    def test_generate_valid_positions_1x2(self):
        box = Box("TestBox", (2, 2))
        block = BasicBlockTypes.blocks["D"]

        assert list(Combinator.generate_valid_positions(block, box))

        expected_coordinates = [np.array((0, 0)), np.array((1, 0)), np.array((0, 0)), np.array((0, 1))]
        expected_orientation = [Orientation.up, Orientation.up, Orientation.right, Orientation.right]
        generated_coordinates = list(Combinator.generate_valid_positions(block, box))

        assert len(expected_coordinates) == len(generated_coordinates)
        for exp_coord, exp_orient, gen_coord in zip(expected_coordinates, expected_orientation, generated_coordinates):
            assert gen_coord.orientation == exp_orient
            assert np.all(exp_coord == gen_coord.coordinate)

    def test_generate_matrix_1x1(self):
        block = BasicBlockTypes.blocks["-"]

        matrix = np.zeros((2, 2))
        Combinator.generate_matrix(matrix, block.tiling[0].space, np.array((0, 0)))
        expected_matrix = np.array(((1, 0), (0, 0)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.generate_matrix(matrix, block.tiling[0].space, np.array((0, 1)))
        expected_matrix = np.array(((0, 1), (0, 0)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.generate_matrix(matrix, block.tiling[0].space, np.array((1, 1)))
        expected_matrix = np.array(((0, 0), (0, 1)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.generate_matrix(matrix, block.tiling[0].space, np.array((1, 0)))
        expected_matrix = np.array(((0, 0), (1, 0)))
        assert np.all(expected_matrix == matrix)

    def test_generate_all_matrix_1x1(self):
        block = BasicBlockTypes.blocks["-"]
        box = Box("Test", (2, 2))
        generate_matrices = list(Combinator.generate_all_matrix(block, box))
        expected_matrices = [np.array(((1, 0), (0, 0))), np.array(((0, 1), (0, 0))), np.array(((0, 0), (0, 1))),
                             np.array(((0, 0), (1, 0)))]

        assert len(generate_matrices) == len(expected_matrices)
        for gen_matrix, exp_matrix in zip(generate_matrices, expected_matrices):
            np.all(gen_matrix == exp_matrix)

    def test_generate_matrix_1x2(self):
        block = BasicBlockTypes.blocks["D"]

        matrix = np.zeros((2, 2))
        Combinator.generate_matrix(matrix, block.tiling[0].space, np.array((0, 0)))
        expected_matrix = np.array(((1, 1), (0, 0)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.generate_matrix(matrix, block.tiling[0].space, np.array((1, 0)))
        expected_matrix = np.array(((0, 0), (1, 1)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.generate_matrix(matrix, block.tiling[1].space, np.array((0, 0)))
        expected_matrix = np.array(((1, 0), (1, 0)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.generate_matrix(matrix, block.tiling[1].space, np.array((0, 1)))
        expected_matrix = np.array(((0, 1), (0, 1)))
        assert np.all(expected_matrix == matrix)

    def test_generate_all_matrix_1x2(self):
        block = BasicBlockTypes.blocks["D"]
        box = Box("Test", (2, 2))
        generate_matrices = list(Combinator.generate_all_matrix(block, box))
        expected_matrices = [np.array(((1, 1), (0, 0))), np.array(((0, 0), (1, 1))), np.array(((1, 0), (1, 0))),
                             np.array(((0, 1), (0, 1)))]

        assert len(generate_matrices) == len(expected_matrices)
        for gen_matrix, exp_matrix in zip(generate_matrices, expected_matrices):
            np.all(gen_matrix == exp_matrix)
    #
    # def test_generate_all_matrix_C(self):
    #     block = BasicBlockTypes.blocks["C"]
    #     box = Box("Test", (2, 2))
    #     generate_matrices = list(Combinator.generate_all_matrix(block, box))
    #     expected_matrices = [np.array(((1, 1), (0, 0))), np.array(((0, 0), (1, 1))), np.array(((1, 0), (1, 0))),
    #                          np.array(((0, 1), (0, 1)))]
    #
    #     assert len(generate_matrices) == len(expected_matrices)
    #     for gen_matrix, exp_matrix in zip(generate_matrices, expected_matrices):
    #         np.all(gen_matrix == exp_matrix)
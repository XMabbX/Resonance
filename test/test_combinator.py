import unittest

import numpy as np

from blocks import Box, BasicBlockTypes, Orientation, Flip
from combinator import Combinator


class TestCombinator(unittest.TestCase):

    def test_generate_coordinates(self):
        box = Box("TestBox", (2, 2))
        self.assertTrue(list(Combinator.generate_coordinate(box)))
        expected_coordinates = [np.array((0, 0)), np.array((0, 1)), np.array((1, 0)), np.array((1, 1))]
        generated_coordinates = list(Combinator.generate_coordinate(box))
        self.assertEqual(len(expected_coordinates), len(generated_coordinates))
        for exp_coord, gen_coord in zip(expected_coordinates, generated_coordinates):
            self.assertTrue(np.all(exp_coord == gen_coord))

    def test_generate_valid_positions_1x1(self):
        box = Box("TestBox", (2, 2))
        block = BasicBlockTypes.blocks["-"]

        assert list(Combinator.generate_valid_positions(block, box))

        expected_coordinates = [np.array((0, 0)), np.array((0, 1)), np.array((1, 0)), np.array((1, 1))]
        generated_coordinates = list(Combinator.generate_coordinate(box))

        assert len(expected_coordinates) == len(generated_coordinates)
        for exp_coord, gen_coord in zip(expected_coordinates, generated_coordinates):
            assert np.all(exp_coord == gen_coord)

    def test_generate_valid_positions_1x2(self):
        box = Box("TestBox", (2, 2))
        block = BasicBlockTypes.blocks["D"]

        assert list(Combinator.generate_valid_positions(block, box))

        expected_coordinates = [np.array((0, 0)), np.array((1, 0)), np.array((0, 0)), np.array((0, 1))]
        expected_orientation = [Orientation.up, Orientation.up, Orientation.right, Orientation.right]
        expected_flip = [Flip.horizontal, Flip.horizontal, Flip.vertical, Flip.vertical]
        generated_coordinates = list(Combinator.generate_valid_positions(block, box))

        assert len(expected_coordinates) == len(generated_coordinates)
        for exp_coord, exp_orient, exp_flip, gen_coord in zip(expected_coordinates, expected_orientation, expected_flip,
                                                              generated_coordinates):
            assert gen_coord.flip == exp_orient
            assert gen_coord.orientation == exp_orient
            assert np.all(exp_coord == gen_coord.coordinate)

    def test_generate_matrix_1x1(self):
        block = BasicBlockTypes.blocks["-"]

        matrix = np.zeros((2, 2))
        Combinator.patch_matrix(matrix, block.tiling[0][0].space, np.array((0, 0)))
        expected_matrix = np.array(((1, 0), (0, 0)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.patch_matrix(matrix, block.tiling[0][0].space, np.array((0, 1)))
        expected_matrix = np.array(((0, 1), (0, 0)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.patch_matrix(matrix, block.tiling[0][0].space, np.array((1, 1)))
        expected_matrix = np.array(((0, 0), (0, 1)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.patch_matrix(matrix, block.tiling[0][0].space, np.array((1, 0)))
        expected_matrix = np.array(((0, 0), (1, 0)))
        assert np.all(expected_matrix == matrix)

    def test_generate_all_matrix_1x1(self):
        block = BasicBlockTypes.blocks["-"]
        box = Box("Test", (2, 2))
        expected_positions = [[0, 0, 0, 0, 0], [1, 0, 1, 0, 0], [2, 1, 0, 0, 0], [3, 1, 1, 0, 0]]
        expected_matrices = [np.array(((1, 0), (0, 0))), np.array(((0, 1), (0, 0))), np.array(((0, 0), (1, 0))),
                             np.array(((0, 0), (0, 1)))]
        idx = 0
        for idx, (position, gen_matrix) in enumerate(Combinator.generate_all_matrix(block, box)):
            self.assertTrue(np.all(gen_matrix == expected_matrices[idx]))
            self.assertTrue(np.all(position == expected_positions[idx]))

        self.assertEqual(idx + 1, len(expected_matrices))

    def test_generate_matrix_1x2(self):
        block = BasicBlockTypes.blocks["D"]

        matrix = np.zeros((2, 2))
        Combinator.patch_matrix(matrix, block.tiling[0][0].space, np.array((0, 0)))
        expected_matrix = np.array(((1, 1), (0, 0)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.patch_matrix(matrix, block.tiling[0][0].space, np.array((1, 0)))
        expected_matrix = np.array(((0, 0), (1, 1)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.patch_matrix(matrix, block.tiling[1][1].space, np.array((0, 0)))
        expected_matrix = np.array(((1, 0), (1, 0)))
        assert np.all(expected_matrix == matrix)

        matrix = np.zeros((2, 2))
        Combinator.patch_matrix(matrix, block.tiling[1][1].space, np.array((0, 1)))
        expected_matrix = np.array(((0, 1), (0, 1)))
        assert np.all(expected_matrix == matrix)

    def test_generate_all_matrix_1x2(self):
        block = BasicBlockTypes.blocks["D"]
        box = Box("Test", (2, 2))
        expected_positions = [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [2, 0, 0, 1, 1], [3, 0, 1, 1, 1]]
        expected_matrices = [np.array(((1, 1), (0, 0))), np.array(((0, 0), (1, 1))), np.array(((1, 0), (1, 0))),
                             np.array(((0, 1), (0, 1)))]
        idx = 0
        for idx, (position, gen_matrix) in enumerate(Combinator.generate_all_matrix(block, box)):
            print(f"Gen position: {position}, expected position: {expected_positions[idx]}")
            self.assertTrue(np.all(gen_matrix == expected_matrices[idx]))

        self.assertEqual(idx + 1, len(expected_matrices))

    def test_generate_valid_positions_C(self):
        box = Box("TestBox", (2, 2))
        block = BasicBlockTypes.blocks["C"]

        assert list(Combinator.generate_valid_positions(block, box))

        expected_coordinates = [np.array((0, 0)), np.array((0, 0)), np.array((0, 0)), np.array((0, 0))]
        expected_orientation = [Orientation.up, Orientation.right, Orientation.down, Orientation.left]
        expected_flip = [Flip.horizontal, Flip.horizontal, Flip.horizontal, Flip.horizontal]
        generated_coordinates = list(Combinator.generate_valid_positions(block, box))

        assert len(expected_coordinates) == len(generated_coordinates)
        for exp_coord, exp_orient, exp_flip, gen_coord in zip(expected_coordinates, expected_orientation, expected_flip,
                                                              generated_coordinates):
            assert gen_coord.flip == exp_flip
            assert gen_coord.orientation == exp_orient
            assert np.all(exp_coord == gen_coord.coordinate)

    def test_generate_all_matrix_C(self):
        block = BasicBlockTypes.blocks["C"]
        box = Box("Test", (2, 2))
        expected_matrices = [np.array(((1, 1), (1, 0))), np.array(((1, 1), (0, 1))), np.array(((0, 1), (1, 1))),
                             np.array(((1, 0), (1, 1)))]
        idx = 0
        for idx, (position, gen_matrix) in enumerate(Combinator.generate_all_matrix(block, box)):
            self.assertTrue(np.all(gen_matrix == expected_matrices[idx]))

        self.assertEqual(idx + 1, len(expected_matrices))

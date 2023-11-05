import unittest

import numpy as np

from blocks import Box, BasicBlockTypes
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

    def test_generate_valid_positions_2x1(self):
        box = Box("TestBox", (2, 2))
        block = BasicBlockTypes.blocks["D"]

        assert list(Combinator.generate_valid_positions(block, box))

        expected_coordinates = [np.array((0, 0)), np.array((0, 1))]
        generated_coordinates = list(Combinator.generate_valid_positions(block, box))

        assert len(expected_coordinates) == len(generated_coordinates)
        for exp_coord, gen_coord in zip(expected_coordinates, generated_coordinates):
            assert np.all(exp_coord == gen_coord.coordinate)

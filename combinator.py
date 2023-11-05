from typing import Iterator

import numpy as np
from nptyping import NDArray

from blocks import Block, Box, Position, Orientation, tCoordinate


class Combinator:

    @classmethod
    def generate_all_matrix(cls, block: Block, box: Box) -> Iterator[NDArray]:
        for coordinate in cls.generate_valid_positions(block, box):
            matrix = np.zeros(box.size)
            yield cls.generate_matrix(matrix, block.tiling[coordinate.orientation].space, coordinate.coordinate)

    @classmethod
    def generate_matrix(cls, matrix: NDArray, patch: NDArray, coordinate: tCoordinate) -> NDArray:
        x = coordinate[0]
        y = coordinate[1]
        matrix[x:x + patch.shape[0], y:y + patch.shape[1]] = patch
        return matrix

    @classmethod
    def generate_valid_positions(cls, block: Block, box: Box) -> Iterator[Position]:
        if block.x == block.y:
            yield from cls._generate_coordinates_for_rotation1(block, box)
        else:
            yield from cls._generate_coordinates_for_rotation1(block, box)
            yield from cls._generate_coordinates_for_rotation2(block, box)

    @classmethod
    def _generate_coordinates_for_rotation1(cls, block: Block, box: Box) -> Iterator[Position]:
        for coordinate in cls.generate_position(box):
            sum_coordinates = coordinate + block.npSize
            if np.all(sum_coordinates <= box.npSize):
                yield Position(block, Orientation.up, coordinate)

    @classmethod
    def _generate_coordinates_for_rotation2(cls, block: Block, box: Box) -> Iterator[Position]:
        for coordinate in cls.generate_position(box):
            sum_coordinates = coordinate + block.npSizeFlip
            if np.all(sum_coordinates <= box.npSize):
                yield Position(block, Orientation.right, coordinate)

    @staticmethod
    def generate_position(box: Box) -> Iterator[tCoordinate]:
        for i in range(box.x):
            for j in range(box.y):
                yield np.array((i, j))

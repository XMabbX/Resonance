from typing import Iterator

import numpy as np
from nptyping import NDArray

from blocks import Block, Box, Position, tCoordinate, Flip


class Combinator:

    @classmethod
    def generate_all_matrix(cls, block: Block, box: Box) -> Iterator[NDArray]:
        for coordinate in cls.generate_valid_positions(block, box):
            matrix = np.zeros(box.size, dtype=int)
            yield cls.generate_matrix(matrix,
                                      block.tiling[coordinate.flip][coordinate.orientation].space,
                                      coordinate.coordinate)

    @classmethod
    def generate_matrix(cls, matrix: NDArray, patch: NDArray, coordinate: tCoordinate) -> NDArray:
        x = coordinate[0]
        y = coordinate[1]
        matrix[x:x + patch.shape[0], y:y + patch.shape[1]] = patch
        return matrix

    @classmethod
    def generate_valid_positions(cls, block: Block, box: Box) -> Iterator[Position]:
        if block.x == block.y:
            yield from cls._generate_positions_for_horizontal(block, box)
        else:
            yield from cls._generate_positions_for_horizontal(block, box)
            yield from cls._generate_positions_for_vertical(block, box)

    @classmethod
    def _generate_positions_for_horizontal(cls, block: Block, box: Box) -> Iterator[Position]:
        for coordinate in cls.generate_coordinate(box):
            sum_coordinates = coordinate + block.npSize
            if np.all(sum_coordinates <= box.npSize):
                for tile in block.tiling[Flip.horizontal].values():
                    yield Position(block, Flip.horizontal, tile.orientation, coordinate)

    @classmethod
    def _generate_positions_for_vertical(cls, block: Block, box: Box) -> Iterator[Position]:
        for coordinate in cls.generate_coordinate(box):
            sum_coordinates = coordinate + block.npSizeFlip
            if np.all(sum_coordinates <= box.npSize):
                for tile in block.tiling[Flip.vertical].values():
                    yield Position(block, Flip.vertical, tile.orientation, coordinate)

    @staticmethod
    def generate_coordinate(box: Box) -> Iterator[tCoordinate]:
        for i in range(box.x):
            for j in range(box.y):
                yield np.array((i, j))

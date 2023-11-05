from typing import Iterator, Annotated, Literal

import numpy as np
import numpy.typing as npt

from blocks import Block, Box, Position, Orientation

tCoordinate = Annotated[npt.NDArray[int], Literal[2]]


class Combinator:

    @classmethod
    def generate_valid_positions(cls, block: Block, box: Box) -> Iterator[Position]:
        for coordinate in cls.generate_position(box):
            sum_coordinates = coordinate + block.npSize
            if np.all(sum_coordinates <= box.npSize):
                yield Position(block, Orientation.up, (coordinate[0], coordinate[1]))

    @staticmethod
    def generate_position(box: Box) -> Iterator[tCoordinate]:
        for i in range(box.x):
            for j in range(box.y):
                yield np.array((i, j))

from typing import List

import numpy as np

import helper as _helper
from numpy.typing import NDArray

from blocks import Box, BasicBlockTypes, tCoordinate
from combinator import Combinator


class TableDrawer:

    @classmethod
    def draw_all_results(cls, results: List[List[int]], position_table: NDArray, box: Box):
        for result in results:
            matrix = np.zeros(box.size, dtype=np.int32)
            print(f"Printing result {result}")
            cls.fill_matrix(result, position_table, matrix, box)
            cls.draw_grid(matrix)

    @classmethod
    def fill_matrix(cls, result: List[int], position_table: NDArray, matrix: NDArray, box: Box):
        temp_matrix = np.zeros(box.size, dtype=np.int32)
        for index in result:
            block_info = position_table[index]
            cls.patch_matrix(temp_matrix, cls.get_block_patch(block_info), block_info[2:4])
            matrix += temp_matrix
            temp_matrix.fill(0)

    @classmethod
    def get_block_patch(cls, block_info: NDArray) -> NDArray:
        block_id: int = block_info[1]
        flip: int = block_info[4]
        orientation = block_info[5]
        block_type = BasicBlockTypes.blocks_ids[block_id]
        block_tilling = block_type.tiling[flip][orientation].space
        return block_tilling * block_type.id

    @classmethod
    def patch_matrix(cls, matrix: NDArray, patch: NDArray, coordinate: tCoordinate) -> NDArray:
        x = coordinate[0]
        y = coordinate[1]
        matrix[x:x + patch.shape[0], y:y + patch.shape[1]] = patch
        return matrix

    @staticmethod
    def _horizontal_sep(x):
        final_string = "+"
        for i in range(x):
            final_string += "--+"
        return final_string

    @classmethod
    def draw_grid(cls, matrix):
        """version with string concatenation"""
        col = matrix.shape[1]
        row = matrix.shape[0]
        final_string = ""
        final_string += cls._horizontal_sep(col) + "\n"
        for i in range(row):
            final_string += "|"
            for j in range(col):
                final_string += f"{matrix[i, j]:02d}" + "|"
            final_string += "\n"
            final_string += cls._horizontal_sep(col) + "\n"
        print(final_string)

    @classmethod
    def print_block(cls, block_name: str, box_size: tuple[int, int]):
        block = BasicBlockTypes.blocks[block_name]
        box = Box("Test", box_size)
        for matrix in Combinator.generate_all_matrix(block, box):
            print(cls.draw_grid(matrix))


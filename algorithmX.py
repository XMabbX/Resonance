from collections import deque

import numpy as np
from nptyping import NDArray

from blocks import Box, Block, tCoordinate
from combinator import Combinator


class AlgorithmX:

    @classmethod
    def patch_matrix(cls, matrix: NDArray, patch: NDArray, coordinate: tCoordinate) -> NDArray:
        x = coordinate[0]
        y = coordinate[1]
        matrix[x:x + patch.shape[0], y:y + patch.shape[1]] = patch
        return matrix

    @classmethod
    def generate_table(cls, box: Box, blocks: list[Block]) -> tuple[NDArray, NDArray]:
        position_rows, table_rows = cls._generate_rows(box, blocks)
        return np.array(position_rows), np.array(table_rows)

    @classmethod
    def _generate_rows(cls, box: Box, blocks: list[Block]):
        table_rows = deque()
        position_rows = deque()
        number_of_blocks = len(blocks)

        for index, block in enumerate(blocks):
            block_identification_array = np.zeros(number_of_blocks, dtype=int)
            block_identification_array[index] = 1
            for position_info, matrix in Combinator.generate_all_matrix(block, box):
                position_rows.append(position_info)
                table_rows.append(np.concatenate((block_identification_array.copy(), matrix.flatten())))
        return position_rows, table_rows

    @classmethod
    def get_cover(cls, matrix: NDArray):
        return cls._get_cover(matrix, np.arange(matrix.shape[0]))

    @classmethod
    def _get_cover(cls, matrix: NDArray, current_indexes: NDArray, level=0):
        print("\t" * level + f"Current level: {level}")
        # print("\t" * level + f"Current matrix:\n {matrix}")

        # If matrix has a column with all 0 return None
        min_index, min_value = cls.get_column_index_lowest_ones(matrix)
        if min_value == 0:
            return None

        found_solutions = []
        # Find rows with ones in the lowest sum column
        for i in cls.get_all_columns_index_with_one_in_index(matrix, min_index):
            current_index = [int(current_indexes[i])]
            print("\t" * level + f"Exploring branch: {current_index[0]}")
            selected_rows = cls.get_all_rows_with_zeros_after_and(matrix, matrix[i])
            new_matrix = matrix[selected_rows][:, matrix[i] == 0]
            if new_matrix.size == 0:
                # If new matrix is empty save current index as solution
                found_solutions.append(current_index)
                continue

            returned_solution = cls._get_cover(new_matrix, current_indexes[selected_rows], level + 1)
            if returned_solution is None:
                # Skip branches with no solution
                continue
            for solution in returned_solution:
                # For each solution create a new branch
                found_solutions.append(current_index + solution)
            print("\t" * level + f"Solutions found for current branch: {found_solutions[-len(returned_solution):]}")

        print("\t" * level + f"Returning solution: {found_solutions}")
        return found_solutions

    @classmethod
    def get_column_index_lowest_ones(cls, matrix: NDArray) -> tuple[int, int]:
        sum_values = np.sum(matrix, axis=0)
        min_index = np.argmin(sum_values)
        return min_index, sum_values[min_index]

    @classmethod
    def get_all_columns_index_with_one_in_index(cls, matrix: NDArray, index: int):
        return np.where(matrix[:, index] == 1)[0]

    @classmethod
    def get_all_rows_with_zeros_after_and(cls, matrix: NDArray, row: NDArray):
        return np.sum(matrix * row, axis=1) == 0

    @classmethod
    def get_rows_index_with_lowest_sum_colum_test(cls, matrix: NDArray):
        min_index, _ = cls.get_column_index_lowest_ones(matrix)
        return cls.get_all_columns_index_with_one_in_index(matrix, min_index)

    @classmethod
    def get_all_row_with_zeros_after_and_using_index_test(cls, matrix: NDArray, index: int):
        return matrix[cls.get_all_rows_with_zeros_after_and(matrix, matrix[index])][:, matrix[index] == 0]

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
        index_count = 0

        for index, block in enumerate(blocks):
            block_identification_array = np.zeros(number_of_blocks, dtype=int)
            block_identification_array[index] = 1
            for position_info, matrix in Combinator.generate_all_matrix(block, box):
                position_rows.append([index_count] + position_info)
                table_rows.append(np.concatenate((block_identification_array.copy(), matrix.flatten())))
                index_count += 1
        return position_rows, table_rows

    @classmethod
    def get_cover(cls, matrix: NDArray, number_of_blocks: int):
        return cls._get_cover(matrix, np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), number_of_blocks)

    @classmethod
    def _get_cover(cls, matrix: NDArray, current_indexes: NDArray, current_blocks: NDArray, number_of_blocks: int,
                   level=0):
        print("\t" * level + f"Current level: {level}")

        remaining_blocks_count = cls.get_columns_index(current_blocks, number_of_blocks)
        found_solutions = []
        for current_column_index in range(remaining_blocks_count):
            print("\t" * level + f"Exploring column: {current_column_index}")
            for i in cls.get_all_rows_index_with_one_in_index(matrix, 0):
                current_index = [int(current_indexes[i])]
                print("\t" * level + f"Exploring branch: {current_index[0]}")
                selected_rows = cls.get_all_rows_with_zeros_after_and(matrix, matrix[i])
                selected_columns = matrix[i] == 0
                selected_blocks = current_blocks[selected_columns]
                full_matrix = np.all(matrix)
                new_matrix = matrix[selected_rows][:, selected_columns]
                if new_matrix.size == 0 and full_matrix:
                    # If new matrix is empty save current index as solution
                    found_solutions.append(current_index)
                    continue

                returned_solution = cls._get_cover(new_matrix, current_indexes[selected_rows], selected_blocks,
                                                   number_of_blocks, level + 1)
                if returned_solution is None:
                    # Skip branches with no solution
                    continue
                for solution in returned_solution:
                    # For each solution create a new branch
                    found_solutions.append(current_index + solution)
                print("\t" * level + f"Solutions found for current branch: {found_solutions[-len(returned_solution):]}")

            # Remove explored column
            selected_rows = np.where(matrix[:, 0] == 0)
            matrix = np.delete(matrix, 0, axis=1)[selected_rows]
            current_blocks = np.delete(current_blocks, 0)
            current_indexes = current_indexes[selected_rows]

        if not found_solutions:
            return None
        # found_solutions = list(
        #     list(other_element) for other_element in set(frozenset(element) for element in found_solutions))
        print("\t" * level + f"Returning solution: {found_solutions}")
        return found_solutions

    @classmethod
    def get_columns_index(cls, current_blocks: NDArray, number_of_blocks):
        return np.where(current_blocks < number_of_blocks)[0].size

    @classmethod
    def get_column_index_lowest_ones(cls, matrix: NDArray) -> tuple[int, int]:
        sum_values = np.sum(matrix, axis=0)
        min_index = np.argmin(sum_values)
        return min_index, sum_values[min_index]

    @classmethod
    def get_all_rows_index_with_one_in_index(cls, matrix: NDArray, index: int):
        return np.where(matrix[:, index] == 1)[0]

    @classmethod
    def get_all_rows_with_zeros_after_and(cls, matrix: NDArray, row: NDArray):
        return np.sum(matrix * row, axis=1) == 0

    @classmethod
    def get_rows_index_with_lowest_sum_colum_test(cls, matrix: NDArray):
        min_index, _ = cls.get_column_index_lowest_ones(matrix)
        return cls.get_all_rows_index_with_one_in_index(matrix, min_index)

    @classmethod
    def get_all_row_with_zeros_after_and_using_index_test(cls, matrix: NDArray, index: int):
        return matrix[cls.get_all_rows_with_zeros_after_and(matrix, matrix[index])][:, matrix[index] == 0]

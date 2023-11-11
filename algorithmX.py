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
    def generate_table(cls, box: Box, blocks: list[Block]) -> NDArray:
        return np.array(cls._generate_rows(box, blocks))

    @classmethod
    def _generate_rows(cls, box: Box, blocks: list[Block]):
        lists = deque()
        number_of_blocks = len(blocks)

        for index, block in enumerate(blocks):
            block_identification_array = np.zeros(number_of_blocks, dtype=int)
            block_identification_array[index] = 1
            for matrix in Combinator.generate_all_matrix(block, box):
                lists.append(np.concatenate((block_identification_array.copy(), matrix.flatten())))
        return lists

    @classmethod
    def get_cover(cls, matrix: NDArray):
        return cls._get_cover(matrix, np.arange(matrix.shape[0]), [])

    @classmethod
    def _get_cover(cls, matrix: NDArray, current_indexes: NDArray, current_solution=None, level=0):
        print(f"Current level: {level} and current solution: {current_solution}")
        # If matrix is empty return None
        min_index, min_value = cls.get_column_index_lowest_ones(matrix)
        if min_value == 0:
            return None

        # Find rows with ones in the lowest sum column
        for i in cls.get_all_columns_index_with_one_in_index(matrix, min_index):
            selected_rows = cls.get_all_rows_with_zeros_after_and(matrix, matrix[i])
            new_matrix = matrix[selected_rows][:, matrix[i] == 0]
            if new_matrix.size == 0:
                current_solution.append([int(current_indexes[i])])
                continue
            returned_solution = cls._get_cover(new_matrix, current_indexes[selected_rows], [current_indexes[i]],
                                               level + 1)
            current_solution.append(returned_solution)

        print(f"Returning solution: {current_solution}")
        return current_solution

    @classmethod
    def unpack_solutions(cls, solutions):
        found_solutions = []
        for sol in solutions:
            if sol is None:
                continue
            found_solutions.append(cls._unpack_list(sol))

        return list(cls._remove_lists(found_solutions))

    @classmethod
    def _remove_lists(cls, elements_list):
        for element in elements_list:
            if isinstance(element[0], list):
                yield from cls._remove_lists(element)
            else:
                yield element

    @classmethod
    def _unpack_list(cls, solutions):
        print(f"Solution: {solutions}")
        if solutions is None or solutions[0] is None:
            raise ValueError(f"Found None in solutions {solutions}")
        if len(solutions) == 1:
            return solutions[0]
        found_inside = []
        current_value = solutions[0]
        for element in solutions[1:]:
            collected = [current_value]
            return_value = cls._unpack_list(element)
            if isinstance(return_value, list):
                for i in return_value:
                    collected.extend(i)
                    found_inside.append(collected)
            elif isinstance(return_value, int):
                collected.append(return_value)
                found_inside.append(collected)
        return found_inside

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

import numpy as np
from nptyping import NDArray

from blocks import Box, Block, tCoordinate


class AlgorithmX:

    @classmethod
    def patch_matrix(cls, matrix: NDArray, patch: NDArray, coordinate: tCoordinate) -> NDArray:
        x = coordinate[0]
        y = coordinate[1]
        matrix[x:x + patch.shape[0], y:y + patch.shape[1]] = patch
        return matrix

    def generate_table(self, box: Box, blocks: list[Block]):
        pass

    @classmethod
    def get_cover(cls, matrix: NDArray):
        return cls._get_cover(matrix, np.arange(matrix.shape[0]), [])

    @classmethod
    def _get_cover(cls, matrix: NDArray, current_indexes: NDArray, current_solution=None, level=0):
        # If matrix is empty return None
        min_index, min_value = cls.get_column_index_lowest_ones(matrix)
        if min_value == 0:
            return None

        # Find rows with ones in the lowest sum column
        for i in cls.get_all_columns_index_with_one_in_index(matrix, min_index):
            selected_rows = cls.get_all_rows_with_zeros_after_and(matrix, matrix[i])
            new_matrix = matrix[selected_rows][:, matrix[i] == 0]
            if new_matrix.size == 0:
                current_solution.append([current_indexes[i]])
                continue
            returned_solution = cls._get_cover(new_matrix, current_indexes[selected_rows], [current_indexes[i]],
                                               level + 1)
            current_solution.append(returned_solution)

        return current_solution

    @classmethod
    def unpack_solutions(cls, solutions):
        found_solutions = []
        for solution in solutions:
            found_sol = tuple(cls._unpack_list(solution))
            if found_sol:
                found_solutions.append(found_sol)
        return found_solutions

    @classmethod
    def _unpack_list(cls, solutions):
        if solutions is None:
            return None
        for solution in solutions:
            if isinstance(solution, list):
                yield from cls._unpack_list(solution)
            else:
                yield solution

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

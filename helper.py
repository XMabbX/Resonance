import numpy as np

from blocks import BasicBlockTypes, Box
from combinator import Combinator


def _horizontal_sep(x):
    final_string = "+"
    for i in range(x):
        final_string += "-+"
    return final_string


def grid(matrix):
    """version with string concatenation"""
    col = matrix.shape[1]
    row = matrix.shape[0]
    final_string = ""
    n = 10
    final_string += _horizontal_sep(col) + "\n"
    for i in range(row):
        final_string += "|"
        for j in range(col):
            final_string += f"{matrix[i, j]}" + "|"
        final_string += "\n"
        final_string += _horizontal_sep(col) + "\n"
    return final_string


def print_block(block_name: str, box_size: tuple[int, int]):
    block = BasicBlockTypes.blocks[block_name]
    box = Box("Test", box_size)
    for matrix in Combinator.generate_all_matrix(block, box):
        print(grid(matrix))


if __name__ == "__main__":
    print(grid(np.array(((1, 0, 5), (0, 1, 3)))))

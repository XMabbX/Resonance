import numpy as np
import dataclasses

from numpy.typing import NDArray

tCoordinate = NDArray


class Orientation:
    up = 0
    right = 1
    down = 2
    left = 3


class Flip:
    horizontal = 0
    vertical = 1


@dataclasses.dataclass
class Tilling:
    orientation: int
    space: NDArray


@dataclasses.dataclass
class Block:
    name: str
    id: int
    size: tuple[int, int]
    number_of_tiles: int
    quantity: int
    tiling: dict[int, dict[int, Tilling]] = None
    stats: None = None
    npSize: np.array = dataclasses.field(init=False)

    def __post_init__(self):
        self.npSize = np.array(self.size)
        self.npSizeFlip = np.flip(self.npSize)

    @property
    def x(self) -> int:
        return self.size[0]

    @property
    def y(self) -> int:
        return self.size[1]


@dataclasses.dataclass
class Position:
    block: Block = dataclasses.field(repr=False)
    flip: int
    orientation: int
    coordinate: tCoordinate


@dataclasses.dataclass
class Box:
    name: str
    size: tuple[int, int]
    npSize: np.array = dataclasses.field(init=False)

    def __post_init__(self):
        self.npSize = np.array(self.size)

    @property
    def x(self) -> int:
        return self.size[0]

    @property
    def y(self) -> int:
        return self.size[1]


class BasicBlockTypes:
    blocks: dict[str, Block] = {
        "-": Block("-", 15, (1, 1), 1, 2, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1,),))),
            }
        }),
        "+": Block("+", 1, (1, 1), 1, 2, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1,),))),
            }
        }),
        "D": Block("D", 2, (1, 2), 2, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1),))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1,), (1,)))),
            }
        }),
        "C": Block("C", 3, (2, 2), 3, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1), (1, 0)))),
                Orientation.right: Tilling(Orientation.right, np.array(((1, 1), (0, 1)))),
                Orientation.down: Tilling(Orientation.down, np.array(((0, 1), (1, 1)))),
                Orientation.left: Tilling(Orientation.left, np.array(((1, 0), (1, 1)))),
            }
        }),
        "I": Block("I", 4, (1, 4), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1, 1, 1),))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1,), (1,), (1,), (1,)))),
            }
        }),
        "T": Block("T", 5, (2, 3), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((0, 1, 0), (1, 1, 1)))),
                Orientation.down: Tilling(Orientation.down, np.array(((1, 1, 1), (0, 1, 0)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1, 0), (1, 1), (1, 0)))),
                Orientation.left: Tilling(Orientation.left, np.array(((0, 1), (1, 1), (0, 1)))),
            }
        }),
        "O": Block("O", 6, (2, 2), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1), (1, 1)))),
            }
        }),
        "S": Block("S", 7, (2, 3), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((0, 1, 1), (1, 1, 0)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1, 0), (1, 1), (0, 1)))),
            }
        }),
        "Z": Block("Z", 8, (2, 3), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1, 0), (0, 1, 1)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((0, 1), (1, 1), (1, 0)))),
            }
        }),
        "L": Block("L", 9, (2, 3), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((0, 0, 1), (1, 1, 1)))),
                Orientation.down: Tilling(Orientation.down, np.array(((1, 1, 1), (1, 0, 0)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1, 0), (1, 0), (1, 1)))),
                Orientation.left: Tilling(Orientation.left, np.array(((1, 1), (0, 1), (0, 1)))),
            }
        }),
        "J": Block("J", 10, (2, 3), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1, 1), (0, 0, 1)))),
                Orientation.down: Tilling(Orientation.down, np.array(((1, 0, 0), (1, 1, 1)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((0, 1), (0, 1), (1, 1)))),
                Orientation.left: Tilling(Orientation.left, np.array(((1, 1), (1, 0), (1, 0)))),
            }
        }),
        "MZ": Block("MZ", 11, (3, 3), 5, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1, 0), (0, 1, 0), (0, 1, 1)))),
                Orientation.right: Tilling(Orientation.right, np.array(((0, 0, 1), (1, 1, 1), (1, 0, 0)))),
            }
        }),
        "MU": Block("MU", 12, (2, 3), 5, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 0, 1), (1, 1, 1)))),
                Orientation.down: Tilling(Orientation.down, np.array(((1, 1, 1), (1, 0, 1)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1, 1), (1, 0), (1, 1)))),
                Orientation.left: Tilling(Orientation.left, np.array(((1, 1), (0, 1), (1, 1)))),
            }
        }),
        "MT": Block("MT", 13, (3, 3), 5, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((0, 1, 0), (0, 1, 0), (1, 1, 1)))),
                Orientation.right: Tilling(Orientation.right, np.array(((1, 0, 0), (1, 1, 1), (1, 0, 0)))),
                Orientation.down: Tilling(Orientation.down, np.array(((1, 1, 1), (0, 1, 0), (0, 1, 0)))),
                Orientation.left: Tilling(Orientation.left, np.array(((0, 0, 1), (1, 1, 1), (0, 0, 1)))),
            }
        }),
        "M+": Block("M+", 14, (3, 3), 5, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))),
            }
        }),
    }

    blocks_ids: dict[int, Block] = {
        15: blocks["-"],
        1: blocks["+"],
        2: blocks["D"],
        3: blocks["C"],
        4: blocks["I"],
        5: blocks["T"],
        6: blocks["O"],
        7: blocks["S"],
        8: blocks["Z"],
        9: blocks["L"],
        10: blocks["J"],
        11: blocks["MZ"],
        12: blocks["MU"],
        13: blocks["MT"],
        14: blocks["M+"],
    }

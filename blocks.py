from typing import Any

import numpy as np
import dataclasses

from nptyping import NDArray, Int32, Shape

tCoordinate = NDArray[Shape['1, 2'], Int32]


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
    space: NDArray[Any, Int32]


@dataclasses.dataclass
class Block:
    name: str
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
        "-": Block("-", (1, 1), 1, 2, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1,),))),
            }
        }),
        "+": Block("+", (1, 1), 1, 2),
        "D": Block("D", (1, 2), 2, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1),))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1,), (1,)))),
            }
        }),
        "C": Block("C", (2, 2), 3, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1), (1, 0)))),
                Orientation.right: Tilling(Orientation.right, np.array(((1, 1), (0, 1)))),
                Orientation.down: Tilling(Orientation.down, np.array(((0, 1), (1, 1)))),
                Orientation.left: Tilling(Orientation.left, np.array(((1, 0), (1, 1)))),
            }
        }),
        "I": Block("I", (1, 4), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1, 1, 1),))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1,), (1,), (1,), (1,)))),
            }
        }),
        "T": Block("T", (2, 3), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((0, 1, 0), (1, 1, 1)))),
                Orientation.down: Tilling(Orientation.down, np.array(((1, 1, 1), (0, 1, 0)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1, 0), (1, 1), (1, 0)))),
                Orientation.left: Tilling(Orientation.left, np.array(((0, 1), (1, 1), (0, 1)))),
            }
        }),
        "O": Block("O", (2, 2), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1), (1, 1)))),
            }
        }),
        "S": Block("S", (2, 3), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((0, 1, 1), (1, 1, 0)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1, 0), (1, 1), (0, 1)))),
            }
        }),
        "Z": Block("Z", (2, 3), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1, 0), (0, 1, 1)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((0, 1), (1, 1), (1, 0)))),
            }
        }),
        "L": Block("L", (2, 3), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((0, 0, 1), (1, 1, 1)))),
                Orientation.down: Tilling(Orientation.down, np.array(((1, 1, 1), (1, 0, 0)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1, 0), (1, 0), (1, 1)))),
                Orientation.left: Tilling(Orientation.left, np.array(((1, 1), (0, 1), (0, 1)))),
            }
        }),
        "J": Block("J", (2, 3), 4, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1, 1), (0, 0, 1)))),
                Orientation.down: Tilling(Orientation.down, np.array(((1, 0, 0), (1, 1, 1)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((0, 1), (0, 1), (1, 1)))),
                Orientation.left: Tilling(Orientation.left, np.array(((1, 1), (1, 0), (1, 0)))),
            }
        }),
        "MZ": Block("MZ", (3, 3), 5, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 1, 0), (0, 1, 0), (0, 1, 1)))),
                Orientation.right: Tilling(Orientation.right, np.array(((0, 0, 1), (1, 1, 1), (1, 0, 0)))),
            }
        }),
        "MU": Block("MU", (2, 3), 5, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((1, 0, 1), (1, 1, 1)))),
                Orientation.down: Tilling(Orientation.down, np.array(((1, 1, 1), (1, 0, 1)))),
            },
            Flip.vertical: {
                Orientation.right: Tilling(Orientation.right, np.array(((1, 1), (1, 0), (1, 1)))),
                Orientation.left: Tilling(Orientation.left, np.array(((1, 1), (0, 1), (1, 1)))),
            }
        }),
        "MT": Block("MT", (3, 3), 5, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((0, 1, 0), (0, 1, 0), (1, 1, 1)))),
                Orientation.right: Tilling(Orientation.right, np.array(((1, 0, 0), (1, 1, 1), (1, 0, 0)))),
                Orientation.down: Tilling(Orientation.down, np.array(((1, 1, 1), (0, 1, 0), (0, 1, 0)))),
                Orientation.left: Tilling(Orientation.left, np.array(((0, 0, 1), (1, 1, 1), (0, 0, 1)))),
            }
        }),
        "M+": Block("M+", (3, 3), 5, 1, tiling={
            Flip.horizontal: {
                Orientation.up: Tilling(Orientation.up, np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))),
            }
        }),
    }

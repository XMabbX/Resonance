import numpy as np
import dataclasses


@dataclasses.dataclass
class Block:
    name: str
    size: tuple[int, int]
    number_of_tiles: int
    quantity: int
    tiling: None = None
    stats: None = None
    npSize: np.array = dataclasses.field(init=False)

    def __post_init__(self):
        self.npSize = np.array(self.size)

    @property
    def x(self) -> int:
        return self.size[0]

    @property
    def y(self) -> int:
        return self.size[1]


class Orientation:
    up = 0
    right = 1
    down = 2
    left = 3


@dataclasses.dataclass
class Position:
    block: Block
    orientation: int
    coordinate: tuple[int, int]


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
        "-": Block("-", (1, 1), 1, 2),
        "+": Block("+", (1, 1), 1, 2),
        "D": Block("D", (2, 1), 2, 1),
        "C": Block("C", (2, 2), 3, 1),
        "I": Block("I", (1, 4), 4, 1),
        "T": Block("T", (3, 2), 4, 1),
        "O": Block("O", (2, 2), 4, 1),
        "S": Block("S", (3, 2), 4, 1),
        "Z": Block("Z", (3, 2), 4, 1),
        "L": Block("L", (2, 3), 4, 1),
        "J": Block("J", (2, 3), 4, 1),
        "MZ": Block("MZ", (3, 3), 5, 1),
        "MU": Block("MU", (3, 2), 5, 1),
        "MT": Block("MT", (3, 3), 5, 1),
        "M+": Block("M+", (3, 3), 5, 1),
    }

import dataclasses

from algorithmX import AlgorithmX
from blocks import Box, Block, BasicBlockTypes


@dataclasses.dataclass
class ResonanceLvl:
    box: Box
    blocks: list[Block]


@dataclasses.dataclass
class Character:
    name: str
    resonance: dict[int, ResonanceLvl]


APPLe = Character("APPLe", {
    1: ResonanceLvl(
        Box("4x4", (4, 4)),
        [
            BasicBlockTypes.blocks["MT"],
            BasicBlockTypes.blocks["D"],
            BasicBlockTypes.blocks["-"],
            BasicBlockTypes.blocks["+"]
        ]
    )
})

if __name__ == "__main__":

    table = AlgorithmX.generate_table(APPLe.resonance[1].box, APPLe.resonance[1].blocks)
    selected_cover = AlgorithmX.get_cover(table)
    unpacked_solutions = AlgorithmX.unpack_solutions(selected_cover)
    print(unpacked_solutions)

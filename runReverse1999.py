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

Leilani = Character("Leilani", {
    2: ResonanceLvl(
        Box("4x4", (4, 4)),
        [
            BasicBlockTypes.blocks["MT"],
            BasicBlockTypes.blocks["L"],
            BasicBlockTypes.blocks["J"],
            BasicBlockTypes.blocks["-"],
            BasicBlockTypes.blocks["-"],
            BasicBlockTypes.blocks["+"],
            BasicBlockTypes.blocks["C"],
            BasicBlockTypes.blocks["D"],
        ]
    )
})

if __name__ == "__main__":
    # table = AlgorithmX.generate_table(APPLe.resonance[1].box, APPLe.resonance[1].blocks)
    # selected_cover = AlgorithmX.get_cover(table)
    # print(selected_cover)

    # position_table, table = AlgorithmX.generate_table(Leilani.resonance[2].box, Leilani.resonance[2].blocks)
    # selected_cover = AlgorithmX.get_cover(table)
    # print(selected_cover)

    # blocks = [BasicBlockTypes.blocks["C"], BasicBlockTypes.blocks["C"]]
    # box = Box("Test", (2,2))
    #
    # position_table, table = AlgorithmX.generate_table(box, blocks)
    # selected_cover = AlgorithmX.get_cover(table)
    # print(selected_cover)

    # blocks = [BasicBlockTypes.blocks["+"]]
    # box = Box("Test", (2, 2))
    #
    # position_table, table = AlgorithmX.generate_table(box, blocks)
    # selected_cover = AlgorithmX.get_cover(table)
    # print(selected_cover)

    blocks = [BasicBlockTypes.blocks["C"], BasicBlockTypes.blocks["-"], BasicBlockTypes.blocks["+"]]
    box = Box("Test", (2, 2))

    position_table, table = AlgorithmX.generate_table(box, blocks)
    print(f"Table: {table}")
    selected_cover = AlgorithmX.get_cover(table)
    print("Solutions")
    print(selected_cover)

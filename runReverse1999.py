import time
import dataclasses

from algorithmX import AlgorithmX
from blocks import Box, Block, BasicBlockTypes
from tableDrawer import TableDrawer


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
    # position_table, table = AlgorithmX.generate_table(APPLe.resonance[1].box, APPLe.resonance[1].blocks)
    # selected_cover = AlgorithmX.get_cover_debug(table, len(APPLe.resonance[1].blocks))
    # print(selected_cover)
    # print(len(selected_cover))
    # TableDrawer.draw_all_results(selected_cover, position_table, APPLe.resonance[1].box)
    start_time = time.time()
    position_table, table = AlgorithmX.generate_table(Leilani.resonance[2].box, Leilani.resonance[2].blocks)
    selected_cover = AlgorithmX.get_cover(table, len(Leilani.resonance[2].blocks))
    # TableDrawer.draw_all_results(selected_cover, position_table, Leilani.resonance[2].box)
    print(f"Time to execute: {time.time() - start_time}")
    # print(selected_cover)
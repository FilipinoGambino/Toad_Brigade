from dataclasses import dataclass, field
from typing import List
from enum import Enum
from .config import EnvConfig
TERM_COLORS = False
try:
    from termcolor import colored
    TERM_COLORS=True
except:
    pass
@dataclass
class FactionInfo:
    color: str = "none"
    alt_color: str = "red"
    faction_id: int = -1

class FactionTypes(Enum):
    Null = FactionInfo(color="gray", faction_id=0)
    AlphaStrike = FactionInfo(color="yellow", faction_id=1)
    MotherMars = FactionInfo(color="green", faction_id=2)
    TheBuilders = FactionInfo(color="blue", faction_id=3)
    FirstMars = FactionInfo(color="red", faction_id=4)

@dataclass
class Player:
    team_id: int
    agent: str
    factories_to_place: int # If tied, player_0's team has this True
    place_first: bool
    robot_count: int = 0
    factory_count: int = 0
    factory_strains: List[int] = field(default_factory=list)
    faction: FactionTypes = None
    water: int = 0
    metal: int = 0
    lichen_count: int = 0
    bid: int = 0


    def state_dict(self):
        return dict(
            team_id=self.team_id,
            faction=self.faction.name,
            # note for optimization, water,metal, factories_to_place doesn't change after the early game.
            water=self.water,
            metal=self.metal,
            factories_to_place=self.factories_to_place,
            factory_strains=self.factory_strains,
            place_first=self.place_first,
        )

    def __str__(self) -> str:
        out = f"[Player {self.team_id}]"
        if TERM_COLORS:
            return colored(out, self.faction.value.color)
        return out
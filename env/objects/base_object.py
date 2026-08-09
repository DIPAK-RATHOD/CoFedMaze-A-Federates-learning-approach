
from dataclasses import dataclass
from typing import Tuple
Coordinate=Tuple[int,int]

@dataclass
class BaseObject:
    object_id:str
    position:Coordinate
    active:bool=True
    def move_to(self, position:Coordinate)->None:
        self.position=position
    def deactivate(self)->None:
        self.active=False
    def activate(self)->None:
        self.active=True

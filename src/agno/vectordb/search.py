from enum import Enum


class SearchType(str, Enum):
    vector = "vector"
    hybrid = "hybrid"

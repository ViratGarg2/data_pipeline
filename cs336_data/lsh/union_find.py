"""Union-Find (disjoint set union) implementation."""

from array import array


class UnionFind:
    """Weighted quick-union with path compression."""

    def __init__(self, n: int):
        # Compact arrays drastically reduce memory for large n.
        self.parent = array("I", range(n))
        self.rank = array("B", [0]) * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

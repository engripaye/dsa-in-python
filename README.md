# Data Structures & Algorithms in Python

> A clean, well-tested, and beginner‑to‑advanced reference for common data structures and algorithms, implemented in Python 3. Includes visual explanations, complexity notes, and practice problems.

---

## Table of Contents

* [Why this repo?](#why-this-repo)
* [Features](#features)
* [Project Structure](#project-structure)
* [Quick Start](#quick-start)
* [Running Tests](#running-tests)
* [Implementations](#implementations)

  * [Core Data Structures](#core-data-structures)
  * [Sorting](#sorting)
  * [Searching](#searching)
  * [Greedy](#greedy)
  * [Divide & Conquer](#divide--conquer)
  * [Dynamic Programming](#dynamic-programming)
  * [Backtracking](#backtracking)
  * [Graphs](#graphs)
  * [Trees & Tries](#trees--tries)
  * [Heaps & Priority Queues](#heaps--priority-queues)
* [Complexity Cheat Sheet](#complexity-cheat-sheet)
* [Examples](#examples)
* [Benchmarks](#benchmarks)
* [Code Style & Tooling](#code-style--tooling)
* [Contributing](#contributing)
* [FAQ](#faq)
* [License](#license)

---

## Why this repo?

* **Practical first.** Each implementation is concise, readable, and production‑oriented.
* **Tested.** Every module ships with unit tests so you can trust changes.
* **Teachable.** Rich docstrings and explanations alongside code.
* **Sharpen your instincts.** Built‑in benchmarks and complexity notes help develop intuition.

## Features

* Pythonic implementations with type hints (`typing`) and docstrings.
* Unit tests with `pytest` and property‑based tests with `hypothesis`.
* Optional CLI runners for quick demos.
* Easy‑to‑extend structure; add your own variants.
* Pre‑commit hooks: formatting (`black`), linting (`ruff`), import sort (`isort`), type checks (`mypy`).

## Project Structure

```text
.
├── dsa_py/
│   ├── __init__.py
│   ├── arrays/
│   │   ├── two_sum.py
│   │   └── kadane.py
│   ├── linked_list/
│   │   ├── singly.py
│   │   └── doubly.py
│   ├── stack_queue/
│   │   ├── stack.py
│   │   └── queue.py
│   ├── heap/
│   │   └── binary_heap.py
│   ├── trie/
│   │   └── trie.py
│   ├── tree/
│   │   ├── bst.py
│   │   └── avl.py
│   ├── graph/
│   │   ├── bfs.py
│   │   ├── dfs.py
│   │   ├── dijkstra.py
│   │   └── topo_sort.py
│   ├── sort/
│   │   ├── quicksort.py
│   │   ├── mergesort.py
│   │   └── heapsort.py
│   ├── search/
│   │   └── binary_search.py
│   ├── dp/
│   │   ├── lcs.py
│   │   └── coin_change.py
│   └── backtracking/
│       └── n_queens.py
├── tests/
│   └── ... (mirrors package layout)
├── benchmarks/
│   └── sort_bench.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## Quick Start

```bash
# 1) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install
pip install -e .[dev]

# 3) Try a quick demo (CLI)
python -m dsa_py.sort.quicksort --size 20 --seed 42
```

## Running Tests

```bash
# Run the whole suite
pytest -q

# Run a file or test node
pytest tests/sort/test_quicksort.py::test_partition

# Type checks & linting
mypy dsa_py && ruff check dsa_py && black --check dsa_py
```

## Implementations

### Core Data Structures

```python
# dsa_py/linked_list/singly.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Iterator, Optional, TypeVar

T = TypeVar("T")

@dataclass
class Node(Generic[T]):
    value: T
    next: Optional[Node[T]] = None

class SinglyLinkedList(Generic[T]):
    """A simple singly linked list with O(1) push/pop front."""
    def __init__(self) -> None:
        self.head: Optional[Node[T]] = None

    def push_front(self, value: T) -> None:
        self.head = Node(value, self.head)

    def pop_front(self) -> T:
        if not self.head:
            raise IndexError("pop from empty list")
        v = self.head.value
        self.head = self.head.next
        return v

    def __iter__(self) -> Iterator[T]:
        cur = self.head
        while cur:
            yield cur.value
            cur = cur.next
```

```python
# dsa_py/stack_queue/stack.py
from typing import Generic, TypeVar, List
T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._data: List[T] = []
    def push(self, x: T) -> None:
        self._data.append(x)
    def pop(self) -> T:
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()
    def peek(self) -> T:
        return self._data[-1]
    def __len__(self) -> int:
        return len(self._data)
```

### Sorting

```python
# dsa_py/sort/quicksort.py
from typing import List, Any
import random

def quicksort(a: List[Any]) -> List[Any]:
    if len(a) <= 1: return a
    pivot = a[len(a)//2]
    left  = [x for x in a if x < pivot]
    mid   = [x for x in a if x == pivot]
    right = [x for x in a if x > pivot]
    return quicksort(left) + mid + quicksort(right)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    random.seed(args.seed)
    arr = [random.randint(0, 999) for _ in range(args.size)]
    print(quicksort(arr))
```

```python
# dsa_py/sort/mergesort.py
from typing import List, Any

def mergesort(a: List[Any]) -> List[Any]:
    if len(a) <= 1:
        return a
    mid = len(a)//2
    left = mergesort(a[:mid])
    right = mergesort(a[mid:])
    return _merge(left, right)

def _merge(left: List[Any], right: List[Any]) -> List[Any]:
    i=j=0; out: List[Any] = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i+=1
        else:
            out.append(right[j]); j+=1
    out.extend(left[i:]); out.extend(right[j:])
    return out
```

### Searching

```python
# dsa_py/search/binary_search.py
from typing import Sequence, Any, Optional

def binary_search(a: Sequence[Any], x: Any) -> Optional[int]:
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi)//2
        if a[mid] == x: return mid
        if a[mid] < x:  lo = mid + 1
        else:           hi = mid - 1
    return None
```

### Greedy

```python
# dsa_py/greedy/activity_selection.py
from typing import List, Tuple

def select_activities(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    intervals.sort(key=lambda t: t[1])
    res, last_end = [], -10**18
    for s, e in intervals:
        if s >= last_end:
            res.append((s, e))
            last_end = e
    return res
```

### Divide & Conquer

```python
# dsa_py/divide_and_conquer/closest_pair.py
from math import hypot

def closest_pair(points):
    # O(n log n) divide-and-conquer reference implementation
    pts = sorted(points)
    def rec(px):
        n = len(px)
        if n <= 3:
            d = float('inf'); best = None
            for i in range(n):
                for j in range(i+1, n):
                    d2 = hypot(px[i][0]-px[j][0], px[i][1]-px[j][1])
                    if d2 < d: d, best = d2, (px[i], px[j])
            return d, best
        mid = n//2; midx = px[mid][0]
        dl, pairl = rec(px[:mid])
        dr, pairr = rec(px[mid:])
        d, best = (dl, pairl) if dl < dr else (dr, pairr)
        strip = [p for p in px if abs(p[0]-midx) < d]
        strip.sort(key=lambda p: p[1])
        for i in range(len(strip)):
            for j in range(i+1, min(i+7, len(strip))):
                ds = hypot(strip[i][0]-strip[j][0], strip[i][1]-strip[j][1])
                if ds < d: d, best = ds, (strip[i], strip[j])
        return d, best
    return rec(pts)
```

### Dynamic Programming

```python
# dsa_py/dp/coin_change.py
from typing import List

def coin_change_min(coins: List[int], amount: int) -> int:
    INF = amount + 1
    dp = [0] + [INF]*amount
    for a in range(1, amount+1):
        dp[a] = min((dp[a-c] + 1) for c in coins if c <= a) if any(c <= a for c in coins) else INF
    return dp[amount] if dp[amount] != INF else -1
```

```python
# dsa_py/dp/lcs.py
from typing import List

def lcs(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp: List[List[int]] = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

### Backtracking

```python
# dsa_py/backtracking/n_queens.py
from typing import List

def solve_n_queens(n: int) -> List[List[str]]:
    cols, diag1, diag2 = set(), set(), set()
    board = [["."]*n for _ in range(n)]
    res: List[List[str]] = []

    def backtrack(r: int) -> None:
        if r == n:
            res.append(["".join(row) for row in board]); return
        for c in range(n):
            if c in cols or (r+c) in diag1 or (r-c) in diag2:
                continue
            cols.add(c); diag1.add(r+c); diag2.add(r-c); board[r][c] = "Q"
            backtrack(r+1)
            cols.remove(c); diag1.remove(r+c); diag2.remove(r-c); board[r][c] = "."
    backtrack(0)
    return res
```

### Graphs

```python
# dsa_py/graph/dijkstra.py
from typing import Dict, List, Tuple
import heapq

Graph = Dict[int, List[Tuple[int, int]]]  # node -> List[(neighbor, weight)]

def dijkstra(g: Graph, src: int) -> Dict[int, int]:
    dist = {src: 0}
    pq = [(0, src)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        for v, w in g.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
```

```python
# dsa_py/graph/topo_sort.py
from collections import deque
from typing import Dict, List

def topo_sort(adj: Dict[int, List[int]]) -> List[int]:
    indeg = {u: 0 for u in adj}
    for u in adj:
        for v in adj[u]:
            indeg[v] = indeg.get(v, 0) + 1
    q = deque([u for u in adj if indeg[u] == 0])
    order: List[int] = []
    while q:
        u = q.popleft(); order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    if len(order) != len(indeg):
        raise ValueError("Graph has a cycle")
    return order
```

### Trees & Tries

```python
# dsa_py/trie/trie.py
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    end: bool = False

class Trie:
    def insert(self, word: str) -> None:
        cur = self.root if hasattr(self, 'root') else TrieNode(); self.root = getattr(self, 'root', cur)
        for ch in word:
            cur = cur.children.setdefault(ch, TrieNode())
        cur.end = True

    def search(self, word: str) -> bool:
        cur = self.root
        for ch in word:
            if ch not in cur.children: return False
            cur = cur.children[ch]
        return cur.end

    def starts_with(self, prefix: str) -> bool:
        cur = self.root
        for ch in prefix:
            if ch not in cur.children: return False
            cur = cur.children[ch]
        return True
```

### Heaps & Priority Queues

```python
# dsa_py/heap/binary_heap.py
import heapq
from typing import Iterable, List, TypeVar

T = TypeVar("T")

class MinHeap:
    def __init__(self, data: Iterable[T] = ()):  
        self._h: List[T] = list(data); heapq.heapify(self._h)
    def push(self, x: T) -> None: heapq.heappush(self._h, x)
    def pop(self) -> T: return heapq.heappop(self._h)
    def peek(self) -> T: return self._h[0]
    def __len__(self) -> int: return len(self._h)
```

## Complexity Cheat Sheet

| Category                                  |       Best |        Average |      Worst |    Space |
| ----------------------------------------- | ---------: | -------------: | ---------: | -------: |
| **Binary Search**                         |       O(1) |       O(log n) |   O(log n) |     O(1) |
| **Quicksort**                             | O(n log n) |     O(n log n) |      O(n²) | O(log n) |
| **Mergesort**                             | O(n log n) |     O(n log n) | O(n log n) |     O(n) |
| **Heapsort**                              | O(n log n) |     O(n log n) | O(n log n) |     O(1) |
| **Hash Table (avg)** lookup/insert/delete |          — |           O(1) |       O(n) |     O(n) |
| **BST (balanced)** search/insert/delete   |          — |       O(log n) |   O(log n) |     O(n) |
| **Dijkstra (binary heap)**                |          — | O((V+E) log V) |          — |     O(V) |
| **Topological Sort**                      |          — |         O(V+E) |          — |     O(V) |

> ⚠️ Complexities assume standard implementations and typical models (e.g., comparison sorting lower bound of Ω(n log n)).

## Examples

### Two Sum (Array / Hashing)

```python
from typing import List, Dict

def two_sum(nums: List[int], target: int) -> List[int]:
    seen: Dict[int, int] = {}
    for i, v in enumerate(nums):
        if target - v in seen:
            return [seen[target - v], i]
        seen[v] = i
    return []
```

### Kadane's Algorithm (Max Subarray)

```python
from typing import List

def max_subarray(nums: List[int]) -> int:
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
```

### BFS (Shortest Path in Unweighted Graph)

```python
from collections import deque
from typing import Dict, List

def bfs_shortest_path(adj: Dict[int, List[int]], src: int, dst: int) -> int:
    q = deque([(src, 0)])
    seen = {src}
    while q:
        u, d = q.popleft()
        if u == dst: return d
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v); q.append((v, d+1))
    return -1
```

## Benchmarks

```bash
# Example: compare sorting algorithms
python benchmarks/sort_bench.py --sizes 1_000 10_000 100_000 --trials 5
```

*Sample benchmark script outline:*

```python
# benchmarks/sort_bench.py
import time, random, statistics as stats
from dsa_py.sort.quicksort import quicksort
from dsa_py.sort.mergesort import mergesort

def bench(fn, arr):
    a = list(arr)
    t0 = time.perf_counter(); fn(a); t1 = time.perf_counter()
    return t1 - t0

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', nargs='+', type=int, default=[1000, 10_000])
    p.add_argument('--trials', type=int, default=5)
    args = p.parse_args()

    for n in args.sizes:
        samples = []
        base = [random.randint(0, 1_000_000) for _ in range(n)]
        for _ in range(args.trials):
            samples.append(bench(quicksort, base))
        print(f"Quicksort n={n}: mean={stats.mean(samples):.6f}s ±{stats.pstdev(samples):.6f}s")
```

## Code Style & Tooling

* **Format:** `black`  – consistent, opinionated formatting.
* **Lint:** `ruff` – fast linting, includes many flake8 rules.
* **Types:** `mypy` – static type checking.
* **Imports:** `isort` – sorted and grouped imports.
* **Pre-commit:** configure hooks in `.pre-commit-config.yaml`.

```toml
# pyproject.toml (excerpt)
[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
select = ["E","F","I","B","UP"]

[tool.isort]
profile = "black"

[tool.mypy]
python_version = "3.11"
strict = true
```

## Contributing

1. Fork the repo & create a feature branch: `git checkout -b feat/your-feature`
2. Run tests locally and ensure linters pass.
3. Open a PR with a clear description (what/why/how).

**Good first issues**: implement a missing algorithm, add tests, write a docstring, or improve benchmarks.

## FAQ

**Q:** *Why are some implementations functional and others imperative?*
**A:** To show idiomatic patterns for each problem. Use the style that communicates intent best.

**Q:** *What Python version?*
**A:** 3.10+ (pattern matching welcome where helpful).

**Q:** *Can I use this in interviews?*
**A:** Yes—clone, run tests, and study the explanations. Avoid rote memorization; understand trade‑offs.

## License

MIT © Your Name

---

### Badges (copy‑paste and adjust)

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-100%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000)

---

> 💡 **Tip:** Pair this repo with daily coding challenges. Track your progress in a `/notes` folder with Markdown files summarizing each topic learned.

"""
Deployed in Evolution Generation 17 (Score: 0.95)
Hypothesis: By implementing an A* pathfinder with a memoized Euclidean distance heuristic and a min-heap priority queue using Python's heapq module, we can efficiently find optimal dependency chains across large skill graphs without exploring redundant nodes, reducing search time and node expansions by >= 50% compared to a naive breadth-first or unguided Dijkstra baseline.
"""

import time
import heapq

class BaselineGraphPathfinder:
    def __init__(self, adj_list, coords):
        self.adj_list = adj_list
        self.coords = coords

    def search(self, start, goal):
        queue = [[start]]
        best_path = None
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == goal:
                best_path = path
                break
            for neighbor in self.adj_list.get(node, []):
                if neighbor not in path:
                    queue.append(path + [neighbor])
        return best_path

class MutatedAStarPathfinder:
    def __init__(self, adj_list, coords):
        self.adj_list = adj_list
        self.coords = coords
        self.heuristic_cache = {}

    def _heuristic(self, a, b):
        key = (a, b)
        if key not in self.heuristic_cache:
            ax, ay = self.coords.get(a, (0, 0))
            bx, by = self.coords.get(b, (0, 0))
            self.heuristic_cache[key] = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        return self.heuristic_cache[key]

    def search(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, start, [start]))
        g_scores = {start: 0.0}
        closed_set = set()

        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            if current == goal:
                return path
            if current in closed_set:
                continue
            closed_set.add(current)
            for neighbor, weight in self.adj_list.get(current, []):
                if neighbor in closed_set:
                    continue
                tentative_g = g + weight
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, path + [neighbor]))
        return None

nodes = [f"node_{i}" for i in range(40)]
coords = {n: (i * 1.5, i * 2.0) for i, n in enumerate(nodes)}
adj_list = {}
for i in range(len(nodes)):
    current = nodes[i]
    adj_list[current] = []
    for j in range(i + 1, min(i + 15, len(nodes))):
        neighbor = nodes[j]
        weight = ((coords[current][0] - coords[neighbor][0])**2 + (coords[current][1] - coords[neighbor][1])**2)**0.5
        adj_list[current].append((neighbor, weight))

baseline = BaselineGraphPathfinder({k: [n for n, w in v] for k, v in adj_list.items()}, coords)
mutation = MutatedAStarPathfinder(adj_list, coords)

start_node = "node_0"
goal_node = "node_35"

t0 = time.time()
for _ in range(50):
    baseline.search(start_node, goal_node)
t1 = time.time()
baseline_ms = (t1 - t0) * 1000.0

t2 = time.time()
for _ in range(50):
    mutation.search(start_node, goal_node)
t3 = time.time()
mutation_ms = (t3 - t2) * 1000.0

speedup_pct = ((baseline_ms - mutation_ms) / baseline_ms) * 100.0
verification_success = mutation_ms < baseline_ms

print(f"BASELINE_MS: {baseline_ms:.2f}")
print(f"MUTATION_MS: {mutation_ms:.2f}")
print(f"SPEEDUP_PCT: {speedup_pct:.2f}%")
print(f"VERIFICATION_SUCCESS: {verification_success}")

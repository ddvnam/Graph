from __future__ import annotations

import heapq
import time
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import (
    DeliveryEnv,
    Order,
    Shipper,
    delivery_reward,
    is_valid_cell,
    r_base,
    valid_next_pos,
)
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
MOVES_WITH_WAIT: Tuple[Move, ...] = ("S", "U", "D", "L", "R")
INF = 10**9


class Constraint:
    """Ràng buộc CBS theo thời gian rời rạc."""

    __slots__ = ("agent", "time", "cell", "kind", "prev")

    def __init__(self, agent: int, time: int, cell: Position, kind: str = "vertex", prev: Optional[Position] = None):
        self.agent = agent
        self.time = time
        self.cell = cell
        self.kind = kind
        self.prev = prev


class CBSNode:
    __slots__ = ("priority", "constraints", "paths")

    def __init__(self, priority: Tuple[int, int, int], constraints: Tuple[Constraint, ...], paths: Dict[int, List[Position]]):
        self.priority = priority
        self.constraints = constraints
        self.paths = paths

    def __lt__(self, other: "CBSNode") -> bool:
        return self.priority < other.priority


class MAPDCBSSolver(Solver):
    """Rolling-horizon MAPD-CBS cho môi trường giao hàng online.

    Solver làm hai việc tại mỗi timestep:
    1. Gán mỗi shipper một mục tiêu hiện tại: delivery nếu đang mang hàng,
       pickup nếu rảnh/còn khả năng chở.
    2. Chạy Conflict-Based Search trên các đường đi ngắn hạn để tránh xung đột
       vertex và edge-swap giữa nhiều shipper. Chỉ bước đầu của kế hoạch được
       thực thi, sau đó solver quan sát lại môi trường và replan.
    """

    method_name = "MAPD-CBS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.N = len(self.grid)
        self.T = int(self.cfg.get("T", 0))
        self._dist_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_cache: Dict[Tuple[Position, Position], Move] = {}
        self.cbs_horizon = 4 if self.N >= 18 else 5
        self.max_cbs_expansions = 16 if self.N >= 18 else 32

        # Online traffic-management state.  These values are updated from the
        # previous action and the current observed position.  They let the
        # solver detect stuck/deadlock situations and make later routes avoid
        # cells that repeatedly caused blocking.
        self._last_positions: Dict[int, Position] = {}
        self._last_actions: Dict[int, Action] = {}
        self._last_desired: Dict[int, Position] = {}
        self._stuck_count: Dict[int, int] = defaultdict(int)
        self._cell_block_heat: Dict[Position, float] = defaultdict(float)
        self._cell_visit_heat: Dict[Position, float] = defaultdict(float)
        self._target_memory: Dict[int, Optional[int]] = {}
        self._target_age: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Grid / shortest-path utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position, allow_wait: bool = False) -> Iterable[Tuple[Move, Position]]:
        moves = MOVES_WITH_WAIT if allow_wait else MOVES
        for move in moves:
            nxt = valid_next_pos(pos, move, self.grid)
            if allow_wait or nxt != pos:
                yield move, nxt

    def _refresh_traffic_state(self, obs: dict) -> None:
        """Update stuck counters and congestion heat from the previous step.

        A shipper is considered blocked when it requested a real move in the
        previous timestep but its current position is unchanged.  The desired
        target cell receives block heat, so future path planning will prefer a
        slightly longer but less congested path.
        """
        shippers: List[Shipper] = obs["shippers"]
        current = {s.id: s.position for s in shippers}

        # Exponential decay keeps the information recent.
        for cell in list(self._cell_block_heat):
            self._cell_block_heat[cell] *= 0.92
            if self._cell_block_heat[cell] < 0.05:
                del self._cell_block_heat[cell]
        for cell in list(self._cell_visit_heat):
            self._cell_visit_heat[cell] *= 0.96
            if self._cell_visit_heat[cell] < 0.05:
                del self._cell_visit_heat[cell]

        for sid, pos in current.items():
            self._cell_visit_heat[pos] += 0.08
            last_pos = self._last_positions.get(sid)
            last_action = self._last_actions.get(sid, ("S", 0))
            last_move = last_action[0] if isinstance(last_action, (tuple, list)) and last_action else "S"
            desired = self._last_desired.get(sid)
            if last_pos is not None and last_move != "S" and pos == last_pos:
                self._stuck_count[sid] += 1
                if desired is not None:
                    self._cell_block_heat[desired] += 2.5
                    self._cell_block_heat[pos] += 0.8
            else:
                self._stuck_count[sid] = 0

        self._last_positions = current

    def _cell_penalty(self, cell: Position) -> float:
        return 1.8 * self._cell_block_heat.get(cell, 0.0) + 0.25 * self._cell_visit_heat.get(cell, 0.0)

    def _bfs_parent(self, start: Position, goal: Position) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None
        q: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}
        while q:
            cur = q.popleft()
            if cur == goal:
                return parent
            for move, nxt in self._neighbors(cur):
                if nxt in parent:
                    continue
                parent[nxt] = (cur, move)
                q.append(nxt)
        return None

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        key = (start, goal)
        if key in self._dist_cache:
            return self._dist_cache[key]
        parent = self._bfs_parent(start, goal)
        if parent is None or goal not in parent:
            self._dist_cache[key] = INF
            return INF
        dist = 0
        cur = goal
        while cur != start:
            prev, _ = parent[cur]
            if prev is None:
                self._dist_cache[key] = INF
                return INF
            cur = prev
            dist += 1
        self._dist_cache[key] = dist
        return dist

    def _move_between(self, a: Position, b: Position) -> Move:
        if a == b:
            return "S"
        for move in MOVES:
            if valid_next_pos(a, move, self.grid) == b:
                return move
        return "S"


    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"
        key = (start, goal)
        if key in self._next_cache:
            return self._next_cache[key]
        parent = self._bfs_parent(start, goal)
        if parent is None or goal not in parent:
            self._next_cache[key] = "S"
            return "S"
        cur = goal
        while True:
            prev, move = parent[cur]
            if prev is None:
                self._next_cache[key] = "S"
                return "S"
            if prev == start:
                self._next_cache[key] = move
                return move
            cur = prev

    def _fast_prioritized_paths(self, starts: Dict[int, Position], goals: Dict[int, Position]) -> Dict[int, List[Position]]:
        occupied_now = set(starts.values())
        reserved_next: Set[Position] = set()
        chosen_next: Dict[int, Position] = {}
        paths: Dict[int, List[Position]] = {}
        for aid in sorted(starts):
            start = starts[aid]
            goal = goals.get(aid, start)
            move = self._next_move(start, goal)
            nxt = valid_next_pos(start, move, self.grid)
            blocked = nxt in reserved_next
            # Tránh đi vào ô hiện tại của agent id nhỏ hơn nếu agent đó không rời đi.
            for other, other_next in chosen_next.items():
                if nxt == starts[other] and other_next == starts[other]:
                    blocked = True
                if nxt == starts[other] and other_next == start:
                    blocked = True
            if blocked:
                nxt = start
            chosen_next[aid] = nxt
            reserved_next.add(nxt)
            paths[aid] = [start, nxt]
        return paths

    # ------------------------------------------------------------------
    # Task scoring / assignment
    # ------------------------------------------------------------------
    def _carried_orders(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]

    def _can_carry_from_state(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        if order.picked or order.delivered:
            return False
        w_now = sum(orders[oid].w for oid in shipper.bag if oid in orders)
        return len(shipper.bag) < shipper.K_max and w_now + order.w <= shipper.W_max

    def _delivery_target(self, shipper: Shipper, orders: Dict[int, Order], t: int) -> Optional[Position]:
        carried = self._carried_orders(shipper, orders)
        if not carried:
            return None

        # Gom theo cùng điểm giao để tận dụng op=2 giao nhiều đơn một lúc.
        groups: Dict[Position, List[Order]] = defaultdict(list)
        for order in carried:
            groups[(order.ex, order.ey)].append(order)

        best_pos: Optional[Position] = None
        best_key: Tuple[float, int, int, int] = (-float("inf"), 0, 0, 0)
        for pos, group in groups.items():
            d = self._distance(shipper.position, pos)
            if d >= INF:
                continue
            eta = min(self.T - 1, t + d)
            value = sum(delivery_reward(order, eta, self.T) for order in group)
            urgent = -min(order.et for order in group)
            priority_sum = sum(order.p for order in group)
            key = (value / (d + 1), len(group), priority_sum, urgent)
            if key > best_key:
                best_key = key
                best_pos = pos
        return best_pos

    def _pickup_score(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int) -> float:
        pickup = (order.sx, order.sy)
        dest = (order.ex, order.ey)
        d1 = self._distance(shipper.position, pickup)
        d2 = self._distance(pickup, dest)
        if d1 >= INF or d2 >= INF:
            return -float("inf")

        eta_delivery = t + d1 + d2
        if eta_delivery >= self.T:
            return -float("inf")

        reward_est = delivery_reward(order, eta_delivery, self.T)
        slack = order.et - eta_delivery
        late_penalty = max(0, -slack) * (0.08 + 0.04 * order.p)
        urgency_bonus = max(0.0, 12.0 / max(slack + 12, 1)) if slack >= 0 else 0.0
        congestion_penalty = 0.15 * self._cell_penalty(pickup) + 0.05 * self._cell_penalty(dest)

        # Bonus nhẹ nếu cùng destination với đơn đang mang; op=2 có thể giao chung.
        batch_bonus = 0.0
        for oid in shipper.bag:
            carried = orders.get(oid)
            if carried and not carried.delivered and (carried.ex, carried.ey) == dest:
                batch_bonus += 4.0

        # Ưu tiên reward trên quãng đường, priority và deadline.
        travel = d1 + d2 + 1
        return reward_est / travel + 1.8 * order.p + urgency_bonus + batch_bonus - late_penalty - congestion_penalty

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved: Set[int],
        t: int,
    ) -> Optional[Order]:
        best: Optional[Order] = None
        best_score = -float("inf")
        for order in orders.values():
            if order.id in reserved:
                continue
            if not self._can_carry_from_state(shipper, order, orders):
                continue
            score = self._pickup_score(shipper, order, orders, t)
            if score == -float("inf"):
                continue
            tie = (-self._distance(shipper.position, (order.sx, order.sy)), order.p, -order.et, -order.id)
            best_tie = (-self._distance(shipper.position, (best.sx, best.sy)), best.p, -best.et, -best.id) if best else None
            if score > best_score or (abs(score - best_score) < 1e-9 and best_tie is not None and tie > best_tie):
                best = order
                best_score = score
            elif best is None:
                best = order
                best_score = score
        return best

    def _assign_targets(self, obs: dict) -> Tuple[Dict[int, Position], Dict[int, Optional[int]]]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs["t"])
        targets: Dict[int, Position] = {}
        target_order: Dict[int, Optional[int]] = {}
        reserved_pickups: Set[int] = set()

        for shipper in sorted(shippers, key=lambda s: s.id):
            # Nếu shipper bị kẹt quá lâu, huỷ target pickup cũ để tránh giữ đơn/đường quá lâu.
            if self._stuck_count.get(shipper.id, 0) >= 8:
                self._target_memory[shipper.id] = None
                self._target_age[shipper.id] = 0

            # Nếu đang ở điểm giao của một hoặc nhiều đơn trong bag, ưu tiên giao ngay.
            deliverable_now = any(
                oid in orders and (orders[oid].ex, orders[oid].ey) == shipper.position and not orders[oid].delivered
                for oid in shipper.bag
            )
            if deliverable_now:
                targets[shipper.id] = shipper.position
                target_order[shipper.id] = None
                continue

            delivery_pos = self._delivery_target(shipper, orders, t)
            if delivery_pos is not None:
                targets[shipper.id] = delivery_pos
                target_order[shipper.id] = None
                continue

            pickup = self._select_pickup(shipper, orders, reserved_pickups, t)
            if pickup is not None:
                reserved_pickups.add(pickup.id)
                targets[shipper.id] = (pickup.sx, pickup.sy)
                target_order[shipper.id] = pickup.id
                if self._target_memory.get(shipper.id) == pickup.id:
                    self._target_age[shipper.id] += 1
                else:
                    self._target_memory[shipper.id] = pickup.id
                    self._target_age[shipper.id] = 1
            else:
                targets[shipper.id] = shipper.position
                target_order[shipper.id] = None

        return targets, target_order

    # ------------------------------------------------------------------
    # CBS low-level and high-level search
    # ------------------------------------------------------------------
    def _constraint_tables(self, constraints: Tuple[Constraint, ...], agent: int):
        vertex: Dict[int, Set[Position]] = defaultdict(set)
        edge: Dict[int, Set[Tuple[Position, Position]]] = defaultdict(set)
        max_t = 0
        for c in constraints:
            if c.agent != agent:
                continue
            max_t = max(max_t, c.time)
            if c.kind == "vertex":
                vertex[c.time].add(c.cell)
            elif c.kind == "edge" and c.prev is not None:
                edge[c.time].add((c.prev, c.cell))
        return vertex, edge, max_t

    def _low_level_plan(
        self,
        agent: int,
        start: Position,
        goal: Position,
        constraints: Tuple[Constraint, ...],
        reserved_vertex: Optional[Dict[int, Set[Position]]] = None,
        reserved_edge: Optional[Dict[int, Set[Tuple[Position, Position]]]] = None,
        extra_avoid: Optional[Set[Position]] = None,
    ) -> Optional[List[Position]]:
        """A* in (cell, time), with CBS constraints and congestion cost."""
        if start == goal:
            return [start]
        if not is_valid_cell(goal, self.grid):
            return None

        dist = self._distance(start, goal)
        if dist >= INF:
            return None

        vertex_cons, edge_cons, max_constraint_t = self._constraint_tables(constraints, agent)
        reserved_vertex = reserved_vertex or {}
        reserved_edge = reserved_edge or {}
        extra_avoid = extra_avoid or set()
        max_time = min(max(self.cbs_horizon + 2, dist + 4, max_constraint_t + 1), max(self.T, 1))

        def h(cell: Position) -> int:
            return self._distance(cell, goal)

        pq: List[Tuple[float, int, Position, int]] = []
        heapq.heappush(pq, (float(h(start)), 0, start, 0))
        best_g: Dict[Tuple[Position, int], float] = {(start, 0): 0.0}
        parent: Dict[Tuple[Position, int], Tuple[Optional[Tuple[Position, int]], Position]] = {
            (start, 0): (None, start)
        }
        counter = 0

        while pq:
            _, _, pos, tm = heapq.heappop(pq)
            g = best_g[(pos, tm)]
            if pos == goal and tm >= max_constraint_t:
                path: List[Position] = []
                state: Optional[Tuple[Position, int]] = (pos, tm)
                while state is not None:
                    path.append(state[0])
                    state = parent[state][0]
                return list(reversed(path))

            if tm >= max_time:
                continue

            for _, nxt in self._neighbors(pos, allow_wait=True):
                nt = tm + 1
                if nxt in vertex_cons.get(nt, set()):
                    continue
                if (pos, nxt) in edge_cons.get(nt, set()):
                    continue
                if nxt in reserved_vertex.get(nt, set()):
                    continue
                if (pos, nxt) in reserved_edge.get(nt, set()):
                    continue

                step_cost = 1.0
                if nxt == pos:
                    step_cost += 0.35
                step_cost += self._cell_penalty(nxt)
                if nxt in extra_avoid and nxt != goal:
                    step_cost += 8.0

                ng = g + step_cost
                state = (nxt, nt)
                if ng >= best_g.get(state, float("inf")):
                    continue
                best_g[state] = ng
                parent[state] = ((pos, tm), pos)
                counter += 1
                heapq.heappush(pq, (ng + h(nxt), counter, nxt, nt))

        return None

    @staticmethod
    def _path_at(path: List[Position], t: int) -> Position:
        if not path:
            raise ValueError("empty path")
        return path[t] if t < len(path) else path[-1]

    def _first_conflict(self, paths: Dict[int, List[Position]]) -> Optional[Tuple[str, int, int, int, Position, Optional[Position]]]:
        ids = sorted(paths)
        for tm in range(1, self.cbs_horizon + 1):
            occupied: Dict[Position, int] = {}
            for aid in ids:
                pos = self._path_at(paths[aid], tm)
                if pos in occupied:
                    return ("vertex", tm, occupied[pos], aid, pos, None)
                occupied[pos] = aid

            for i, a in enumerate(ids):
                a_prev, a_now = self._path_at(paths[a], tm - 1), self._path_at(paths[a], tm)
                for b in ids[i + 1 :]:
                    b_prev, b_now = self._path_at(paths[b], tm - 1), self._path_at(paths[b], tm)
                    if a_prev == b_now and b_prev == a_now and a_prev != a_now:
                        return ("edge", tm, a, b, a_now, a_prev)
        return None

    def _paths_cost(self, paths: Dict[int, List[Position]]) -> int:
        return sum(max(0, len(path) - 1) for path in paths.values())

    def _cbs_plan(self, starts: Dict[int, Position], goals: Dict[int, Position]) -> Dict[int, List[Position]]:
        constraints: Tuple[Constraint, ...] = tuple()
        root_paths: Dict[int, List[Position]] = {}
        for aid in sorted(starts):
            path = self._low_level_plan(aid, starts[aid], goals[aid], constraints)
            root_paths[aid] = path if path is not None else [starts[aid]]

        root = CBSNode((0, self._paths_cost(root_paths), 0), constraints, root_paths)
        open_list: List[CBSNode] = [root]
        expansions = 0

        while open_list and expansions < self.max_cbs_expansions:
            node = heapq.heappop(open_list)
            expansions += 1
            conflict = self._first_conflict(node.paths)
            if conflict is None:
                return node.paths

            kind, tm, a1, a2, cell, prev = conflict
            for aid in (a1, a2):
                if kind == "vertex":
                    new_constraint = Constraint(agent=aid, time=tm, cell=cell, kind="vertex")
                else:
                    # Với edge conflict, ràng buộc đúng chiều di chuyển của từng agent.
                    p0 = self._path_at(node.paths[aid], tm - 1)
                    p1 = self._path_at(node.paths[aid], tm)
                    new_constraint = Constraint(agent=aid, time=tm, cell=p1, kind="edge", prev=p0)

                new_constraints = node.constraints + (new_constraint,)
                new_paths = dict(node.paths)
                replanned = self._low_level_plan(aid, starts[aid], goals[aid], new_constraints)
                if replanned is None:
                    continue
                new_paths[aid] = replanned
                conflicts_left = 1 if self._first_conflict(new_paths) else 0
                heapq.heappush(
                    open_list,
                    CBSNode((conflicts_left, self._paths_cost(new_paths), expansions), new_constraints, new_paths),
                )

        return self._prioritized_plan(starts, goals)

    def _prioritized_plan(self, starts: Dict[int, Position], goals: Dict[int, Position]) -> Dict[int, List[Position]]:
        """Reservation-table fallback.

        Agents are planned sequentially.  Later agents avoid the vertex and
        edge reservations of earlier agents, which directly matches the env
        rule that lower id shippers win cell conflicts.  Stuck agents receive
        a small priority boost so they can escape local deadlocks.
        """
        paths: Dict[int, List[Position]] = {}
        reserved_vertex: Dict[int, Set[Position]] = defaultdict(set)
        reserved_edge: Dict[int, Set[Tuple[Position, Position]]] = defaultdict(set)

        def priority(aid: int) -> Tuple[int, int]:
            # Lower tuple plans first.  Keep id as dominant tie-break to remain
            # consistent with env collision priority, but lift badly stuck agents.
            return (-min(self._stuck_count.get(aid, 0), 12), aid)

        for aid in sorted(starts, key=priority):
            start = starts[aid]
            goal = goals.get(aid, start)
            extra_avoid = {cell for cell, heat in self._cell_block_heat.items() if heat >= 5.0}
            if start in extra_avoid:
                extra_avoid.remove(start)
            path = self._low_level_plan(
                aid, start, goal, tuple(), reserved_vertex=reserved_vertex, reserved_edge=reserved_edge, extra_avoid=extra_avoid
            )
            if path is None:
                # Retry without strong avoid cells; reaching the target is still
                # better than waiting forever if the corridor is unavoidable.
                path = self._low_level_plan(aid, start, goal, tuple(), reserved_vertex=reserved_vertex, reserved_edge=reserved_edge)
            if path is None:
                path = [start]
            paths[aid] = path

            for tm in range(1, self.cbs_horizon + 2):
                pos = self._path_at(path, tm)
                prev = self._path_at(path, tm - 1)
                reserved_vertex[tm].add(pos)
                # Prevent another agent from taking the opposite edge at the
                # same time, which is the classic swap deadlock in corridors.
                reserved_edge[tm].add((pos, prev))
        return paths

    def _safe_first_step_plan(self, starts: Dict[int, Position], goals: Dict[int, Position]) -> Dict[int, List[Position]]:
        """One-step conflict repair used after CBS/prioritized planning.

        It handles the exact env semantics: shippers are processed by id, and
        a shipper is blocked if its target cell is still occupied.  This repair
        converts unsafe first moves into waits or alternate sidesteps.
        """
        paths = self._prioritized_plan(starts, goals)
        occupied = set(starts.values())
        chosen: Dict[int, Position] = {}
        fixed: Dict[int, List[Position]] = {}
        for aid in sorted(starts):
            start = starts[aid]
            path = paths.get(aid, [start])
            desired = self._path_at(path, 1) if len(path) > 1 else start

            occupied.discard(start)
            unsafe = desired in occupied or desired in chosen.values()
            for oid, onext in chosen.items():
                if desired == starts[oid] and onext == start:
                    unsafe = True
                    break

            if unsafe:
                # Try a low-heat sidestep that does not create immediate conflict.
                candidates: List[Tuple[float, Position]] = [(3.0 + self._cell_penalty(start), start)]
                for _, nxt in self._neighbors(start, allow_wait=False):
                    if nxt in occupied or nxt in chosen.values():
                        continue
                    swap = any(nxt == starts[oid] and onext == start for oid, onext in chosen.items())
                    if swap:
                        continue
                    candidates.append((self._cell_penalty(nxt) + self._distance(nxt, goals.get(aid, start)), nxt))
                desired = min(candidates, key=lambda x: x[0])[1]

            chosen[aid] = desired
            occupied.add(desired)
            fixed[aid] = [start, desired]
        return fixed

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------
    def _has_delivery_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        return any(
            oid in orders and not orders[oid].delivered and (orders[oid].ex, orders[oid].ey) == pos
            for oid in shipper.bag
        )

    def _can_pick_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        for order in orders.values():
            if order.picked or order.delivered:
                continue
            if (order.sx, order.sy) != pos:
                continue
            if self._can_carry_from_state(shipper, order, orders):
                return True
        return False

    def _actions_from_paths(
        self,
        obs: dict,
        paths: Dict[int, List[Position]],
        target_order: Dict[int, Optional[int]],
    ) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        actions: Dict[int, Action] = {}

        for shipper in sorted(shippers, key=lambda s: s.id):
            path = paths.get(shipper.id, [shipper.position])
            next_pos = self._path_at(path, 1) if len(path) > 1 else shipper.position
            move = self._move_between(shipper.position, next_pos)

            # Delivery ưu tiên hơn pickup vì op=2 có thể giao nhiều đơn chung đích.
            if self._has_delivery_at(shipper, orders, next_pos):
                actions[shipper.id] = (move, 2)
                continue

            oid = target_order.get(shipper.id)
            if oid is not None:
                order = orders.get(oid)
                if order and not order.picked and not order.delivered and (order.sx, order.sy) == next_pos:
                    actions[shipper.id] = (move, 1)
                    continue

            # Nếu CBS buộc shipper đi lệch nhưng vẫn tới ô có đơn khả thi, nhặt 1 đơn tốt nhất.
            if not shipper.bag and self._can_pick_at(shipper, orders, next_pos):
                actions[shipper.id] = (move, 1)
            else:
                actions[shipper.id] = (move, 0)
        return actions

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        self._refresh_traffic_state(obs)
        shippers: List[Shipper] = obs["shippers"]
        targets, target_order = self._assign_targets(obs)
        starts = {shipper.id: shipper.position for shipper in shippers}

        # CBS is useful on small/medium maps.  On large maps the reservation
        # planner is faster and, with congestion heat, avoids the C5/C6 style
        # long bottleneck deadlocks better than a tiny CBS horizon.
        if self.N <= 15:
            paths = self._cbs_plan(starts, targets)
        else:
            paths = self._safe_first_step_plan(starts, targets)

        actions = self._actions_from_paths(obs, paths, target_order)
        self._last_actions = actions
        self._last_desired = {
            sid: valid_next_pos(starts[sid], actions.get(sid, ("S", 0))[0], self.grid)
            for sid in starts
        }
        return actions

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()
        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)

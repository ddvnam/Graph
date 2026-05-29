from __future__ import annotations
import heapq
import time
from collections import OrderedDict, defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set, Tuple
from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, r_base, valid_next_pos
from solvers.solver import Solver
Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]
MOVES: Tuple[Move, ...] = ('U', 'D', 'L', 'R')
MOVES_WITH_WAIT: Tuple[Move, ...] = ('S', 'U', 'D', 'L', 'R')
INF = 10 ** 9

class Constraint:
    __slots__ = ('agent', 'time', 'cell', 'kind', 'prev')

    def __init__(self, agent: int, time: int, cell: Position, kind: str='vertex', prev: Optional[Position]=None):
        self.agent = agent
        self.time = time
        self.cell = cell
        self.kind = kind
        self.prev = prev

class CBSNode:
    __slots__ = ('priority', 'constraints', 'paths')

    def __init__(self, priority: Tuple[int, int, int], constraints: Tuple[Constraint, ...], paths: Dict[int, List[Position]]):
        self.priority = priority
        self.constraints = constraints
        self.paths = paths

    def __lt__(self, other: 'CBSNode') -> bool:
        return self.priority < other.priority

class MAPDCBSSolver(Solver):
    method_name = 'MAPD-CBS'
    FAST_PLANNER_N = 50
    MAX_ORDERS_PER_SHIPPER = 150
    DIST_MAP_CACHE_LIMIT = 512

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        init_obs = env.observe()
        self.grid = init_obs['grid']
        self.N = int(init_obs['N'])
        self.T = int(init_obs['T'])
        self._dist_to_goal_cache: OrderedDict[Position, Dict[Position, int]] = OrderedDict()
        self._next_cache: Dict[Tuple[Position, Position], Move] = {}
        self._order_trip_dist: Dict[int, Tuple[Position, Position, int]] = {}
        self._waiting_by_pickup: Dict[Position, List[Order]] = defaultdict(list)
        self.cbs_horizon = 4 if self.N >= 18 else 5
        self.max_cbs_expansions = 16 if self.N >= 18 else 32
        self._last_positions: Dict[int, Position] = {}
        self._last_actions: Dict[int, Action] = {}
        self._last_desired: Dict[int, Position] = {}
        self._stuck_count: Dict[int, int] = defaultdict(int)
        self._cell_block_heat: Dict[Position, float] = defaultdict(float)
        self._cell_visit_heat: Dict[Position, float] = defaultdict(float)
        self._target_memory: Dict[int, Optional[int]] = {}
        self._target_age: Dict[int, int] = defaultdict(int)
        self._idle_count: Dict[int, int] = defaultdict(int)
        self._order_claim: Dict[int, int] = {}
        self._claim_age: Dict[int, int] = defaultdict(int)

    def _neighbors(self, pos: Position, allow_wait: bool=False) -> Iterable[Tuple[Move, Position]]:
        moves = MOVES_WITH_WAIT if allow_wait else MOVES
        for move in moves:
            nxt = valid_next_pos(pos, move, self.grid)
            if allow_wait or nxt != pos:
                yield (move, nxt)

    def _refresh_traffic_state(self, obs: dict) -> None:
        shippers: List[Shipper] = obs['shippers']
        current = {s.id: s.position for s in shippers}
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
            last_action = self._last_actions.get(sid, ('S', 0))
            last_move = last_action[0] if isinstance(last_action, (tuple, list)) and last_action else 'S'
            desired = self._last_desired.get(sid)
            last_op = last_action[1] if isinstance(last_action, (tuple, list)) and len(last_action) >= 2 else 0
            if last_pos is not None and last_move != 'S' and (pos == last_pos):
                self._stuck_count[sid] += 1
                self._idle_count[sid] = 0
                if desired is not None:
                    self._cell_block_heat[desired] += 2.5
                    self._cell_block_heat[pos] += 0.8
            else:
                self._stuck_count[sid] = 0
                if last_pos is not None and pos == last_pos and (last_move == 'S') and (last_op == 0):
                    self._idle_count[sid] += 1
                else:
                    self._idle_count[sid] = 0
        self._last_positions = current

    def _cell_penalty(self, cell: Position) -> float:
        return 1.8 * self._cell_block_heat.get(cell, 0.0) + 0.25 * self._cell_visit_heat.get(cell, 0.0)

    def _bfs_parent(self, start: Position, goal: Position) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None
        q: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, 'S')}
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

    def _get_dist_map(self, goal: Position) -> Dict[Position, int]:
        cached = self._dist_to_goal_cache.get(goal)
        if cached is not None:
            self._dist_to_goal_cache.move_to_end(goal)
            return cached
        if not is_valid_cell(goal, self.grid):
            self._dist_to_goal_cache[goal] = {}
            self._dist_to_goal_cache.move_to_end(goal)
            while len(self._dist_to_goal_cache) > self.DIST_MAP_CACHE_LIMIT:
                self._dist_to_goal_cache.popitem(last=False)
            return self._dist_to_goal_cache[goal]
        q: deque[Position] = deque([goal])
        dist: Dict[Position, int] = {goal: 0}
        while q:
            cur = q.popleft()
            nd = dist[cur] + 1
            for _, nxt in self._neighbors(cur, allow_wait=False):
                if nxt in dist:
                    continue
                dist[nxt] = nd
                q.append(nxt)
        self._dist_to_goal_cache[goal] = dist
        self._dist_to_goal_cache.move_to_end(goal)
        while len(self._dist_to_goal_cache) > self.DIST_MAP_CACHE_LIMIT:
            self._dist_to_goal_cache.popitem(last=False)
        return dist

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        return self._get_dist_map(goal).get(start, INF)

    def _order_trip_distance(self, order: Order) -> int:
        pickup = (order.sx, order.sy)
        dest = (order.ex, order.ey)
        cached = self._order_trip_dist.get(order.id)
        if cached is not None and cached[0] == pickup and (cached[1] == dest):
            return cached[2]
        d = self._distance(pickup, dest)
        self._order_trip_dist[order.id] = (pickup, dest, d)
        return d

    def _move_between(self, a: Position, b: Position) -> Move:
        if a == b:
            return 'S'
        for move in MOVES:
            if valid_next_pos(a, move, self.grid) == b:
                return move
        return 'S'

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return 'S'
        key = (start, goal)
        if key in self._next_cache:
            return self._next_cache[key]
        dist_map = self._get_dist_map(goal)
        cur_d = dist_map.get(start, INF)
        if cur_d >= INF:
            self._next_cache[key] = 'S'
            return 'S'
        best_move = 'S'
        best_key = (cur_d, self._cell_penalty(start), 'S')
        for move, nxt in self._neighbors(start, allow_wait=False):
            d = dist_map.get(nxt, INF)
            if d >= INF:
                continue
            cand = (d, self._cell_penalty(nxt), move)
            if cand < best_key:
                best_key = cand
                best_move = move
        self._next_cache[key] = best_move
        return best_move

    def _fast_prioritized_paths(self, starts: Dict[int, Position], goals: Dict[int, Position]) -> Dict[int, List[Position]]:
        occupied = set(starts.values())
        chosen_next: Dict[int, Position] = {}
        paths: Dict[int, List[Position]] = {}
        for aid in sorted(starts):
            start = starts[aid]
            goal = goals.get(aid, start)
            move = self._next_move(start, goal)
            desired = valid_next_pos(start, move, self.grid)
            occupied.discard(start)
            unsafe = desired in occupied or desired in chosen_next.values()
            for oid, onext in chosen_next.items():
                if desired == starts[oid] and onext == start:
                    unsafe = True
                    break
            if unsafe:
                dist_map = self._get_dist_map(goal)
                candidates: List[Tuple[float, Position]] = [(3.0 + self._cell_penalty(start), start)]
                for _, nb in self._neighbors(start, allow_wait=False):
                    if nb in occupied or nb in chosen_next.values():
                        continue
                    if any((nb == starts[oid] and onext == start for oid, onext in chosen_next.items())):
                        continue
                    candidates.append((dist_map.get(nb, INF) + self._cell_penalty(nb), nb))
                desired = min(candidates, key=lambda x: x[0])[1]
            chosen_next[aid] = desired
            occupied.add(desired)
            paths[aid] = [start, desired]
        return paths

    def _carried_orders(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [orders[oid] for oid in shipper.bag if oid in orders and (not orders[oid].delivered)]

    def _can_carry_from_state(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        if order.picked or order.delivered:
            return False
        w_now = sum((orders[oid].w for oid in shipper.bag if oid in orders))
        return len(shipper.bag) < shipper.K_max and w_now + order.w <= shipper.W_max

    def _delivery_target(self, shipper: Shipper, orders: Dict[int, Order], t: int) -> Optional[Position]:
        carried = self._carried_orders(shipper, orders)
        if not carried:
            return None
        groups: Dict[Position, List[Order]] = defaultdict(list)
        for order in carried:
            groups[order.ex, order.ey].append(order)
        best_pos: Optional[Position] = None
        best_key: Tuple[float, int, int, int] = (-float('inf'), 0, 0, 0)
        for pos, group in groups.items():
            d = self._distance(shipper.position, pos)
            if d >= INF:
                continue
            eta = min(self.T - 1, t + d)
            value = sum((delivery_reward(order, eta, self.T) for order in group))
            urgent = -min((order.et for order in group))
            priority_sum = sum((order.p for order in group))
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
            return -float('inf')
        eta_delivery = t + d1 + d2
        if eta_delivery >= self.T:
            return -float('inf')
        reward_est = delivery_reward(order, eta_delivery, self.T)
        slack = order.et - eta_delivery
        late_penalty = max(0, -slack) * (0.08 + 0.04 * order.p)
        urgency_bonus = max(0.0, 12.0 / max(slack + 12, 1)) if slack >= 0 else 0.0
        congestion_penalty = 0.15 * self._cell_penalty(pickup) + 0.05 * self._cell_penalty(dest)
        batch_bonus = 0.0
        for oid in shipper.bag:
            carried = orders.get(oid)
            if carried and (not carried.delivered) and ((carried.ex, carried.ey) == dest):
                batch_bonus += 4.0
        travel = d1 + d2 + 1
        return reward_est / travel + 1.8 * order.p + urgency_bonus + batch_bonus - late_penalty - congestion_penalty

    def _select_pickup(self, shipper: Shipper, orders: Dict[int, Order], reserved: Set[int], t: int) -> Optional[Order]:
        best: Optional[Order] = None
        best_score = -float('inf')
        for order in orders.values():
            if order.id in reserved:
                continue
            if not self._can_carry_from_state(shipper, order, orders):
                continue
            score = self._pickup_score(shipper, order, orders, t)
            if score == -float('inf'):
                continue
            tie = (-self._distance(shipper.position, (order.sx, order.sy)), order.p, -order.et, -order.id)
            best_tie = (-self._distance(shipper.position, (best.sx, best.sy)), best.p, -best.et, -best.id) if best else None
            if score > best_score or (abs(score - best_score) < 1e-09 and best_tie is not None and (tie > best_tie)):
                best = order
                best_score = score
            elif best is None:
                best = order
                best_score = score
        return best

    def _active_order_cells(self, orders: Dict[int, Order]) -> Set[Position]:
        cells: Set[Position] = set()
        for order in orders.values():
            if order.delivered:
                continue
            if not order.picked:
                cells.add((order.sx, order.sy))
            cells.add((order.ex, order.ey))
        return cells

    def _cell_degree(self, cell: Position) -> int:
        return sum((1 for _ in self._neighbors(cell, allow_wait=False)))

    def _idle_parking_target(self, shipper: Shipper, shippers: List[Shipper], orders: Dict[int, Order], reserved_targets: Set[Position]) -> Position:
        start = shipper.position
        occupied = {s.position for s in shippers if s.id != shipper.id}
        active_cells = self._active_order_cells(orders)

        def parking_penalty(cell: Position, is_wait: bool) -> float:
            penalty = self._cell_penalty(cell)
            if cell in active_cells:
                penalty += 30.0
            if cell in reserved_targets:
                penalty += 12.0
            if cell in occupied:
                penalty += 1000.0
            degree = self._cell_degree(cell)
            if degree <= 1:
                penalty += 4.0
            elif degree == 2:
                penalty += 2.0
            if is_wait:
                penalty += 0.4
            return penalty
        current_penalty = parking_penalty(start, True)
        must_move = start in active_cells or current_penalty >= 6.0 or self._idle_count.get(shipper.id, 0) >= 8
        candidates: List[Tuple[float, Position]] = []
        if not must_move:
            candidates.append((current_penalty, start))
        else:
            candidates.append((current_penalty + 8.0, start))
        for _, nb in self._neighbors(start, allow_wait=False):
            if nb in occupied:
                continue
            candidates.append((parking_penalty(nb, False) + 0.5, nb))
        if not candidates:
            return start
        return min(candidates, key=lambda x: x[0])[1]

    def _refresh_order_claims(self, orders: Dict[int, Order]) -> None:
        live_waiting = {oid for oid, o in orders.items() if not o.picked and (not o.delivered)}
        for oid in list(self._order_claim):
            if oid not in live_waiting:
                del self._order_claim[oid]
                self._claim_age.pop(oid, None)
        for oid in list(self._order_trip_dist):
            if oid not in orders or orders[oid].delivered:
                self._order_trip_dist.pop(oid, None)

    def _build_waiting_index(self, orders: Dict[int, Order]) -> List[Order]:
        self._waiting_by_pickup = defaultdict(list)
        waiting_orders: List[Order] = []
        for order in orders.values():
            if order.picked or order.delivered:
                continue
            waiting_orders.append(order)
            self._waiting_by_pickup[order.sx, order.sy].append(order)
        return waiting_orders

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _assign_targets(self, obs: dict) -> Tuple[Dict[int, Position], Dict[int, Optional[int]]]:
        orders: Dict[int, Order] = obs['orders']
        shippers: List[Shipper] = obs['shippers']
        t = int(obs['t'])
        targets: Dict[int, Position] = {}
        target_order: Dict[int, Optional[int]] = {}
        free_shippers: List[Shipper] = []
        reserved_targets: Set[Position] = set()
        self._refresh_order_claims(orders)
        waiting_orders = self._build_waiting_index(orders)
        for shipper in sorted(shippers, key=lambda s: s.id):
            if self._stuck_count.get(shipper.id, 0) >= 8:
                old_oid = self._target_memory.get(shipper.id)
                if old_oid is not None and self._order_claim.get(old_oid) == shipper.id:
                    self._order_claim.pop(old_oid, None)
                    self._claim_age.pop(old_oid, None)
                self._target_memory[shipper.id] = None
                self._target_age[shipper.id] = 0
            deliverable_now = any((oid in orders and (orders[oid].ex, orders[oid].ey) == shipper.position and (not orders[oid].delivered) for oid in shipper.bag))
            if deliverable_now:
                targets[shipper.id] = shipper.position
                target_order[shipper.id] = None
                reserved_targets.add(shipper.position)
                continue
            delivery_pos = self._delivery_target(shipper, orders, t)
            if delivery_pos is not None:
                targets[shipper.id] = delivery_pos
                target_order[shipper.id] = None
                reserved_targets.add(delivery_pos)
                continue
            free_shippers.append(shipper)
        candidates: List[Tuple[float, int, int, int, int, int]] = []
        for shipper in free_shippers:
            if not waiting_orders:
                break
            if len(waiting_orders) <= self.MAX_ORDERS_PER_SHIPPER:
                candidate_orders = list(waiting_orders)
            else:
                s_pos = shipper.position
                candidate_orders = heapq.nsmallest(self.MAX_ORDERS_PER_SHIPPER, waiting_orders, key=lambda order: (self._manhattan(s_pos, (order.sx, order.sy)), -order.p, order.et, order.id))
                claimed_ids = [oid for oid, sid in self._order_claim.items() if sid == shipper.id]
                if claimed_ids:
                    seen = {o.id for o in candidate_orders}
                    for oid in claimed_ids:
                        order = orders.get(oid)
                        if order is not None and (not order.picked) and (not order.delivered) and (oid not in seen):
                            candidate_orders.append(order)
                            seen.add(oid)
            for order in candidate_orders:
                if not self._can_carry_from_state(shipper, order, orders):
                    continue
                pickup = (order.sx, order.sy)
                dest = (order.ex, order.ey)
                d1 = self._distance(shipper.position, pickup)
                d2 = self._order_trip_distance(order)
                if d1 >= INF or d2 >= INF:
                    continue
                eta_delivery = t + d1 + d2
                if eta_delivery >= self.T:
                    continue
                reward_est = delivery_reward(order, eta_delivery, self.T)
                slack = order.et - eta_delivery
                late_penalty = max(0, -slack) * (0.08 + 0.04 * order.p)
                urgency_bonus = max(0.0, 12.0 / max(slack + 12, 1)) if slack >= 0 else 0.0
                congestion_penalty = 0.15 * self._cell_penalty(pickup) + 0.05 * self._cell_penalty(dest)
                score = reward_est / (d1 + d2 + 1) - 2.2 * d1 - 0.4 * d2 + 2.0 * order.p + urgency_bonus - late_penalty - congestion_penalty
                claim_owner = self._order_claim.get(order.id)
                if claim_owner == shipper.id and self._stuck_count.get(shipper.id, 0) < 6:
                    score += min(18.0, 8.0 + 1.5 * self._claim_age.get(order.id, 0))
                elif claim_owner is not None and claim_owner != shipper.id:
                    owner_stuck = self._stuck_count.get(claim_owner, 0)
                    if owner_stuck < 6:
                        score -= 10.0
                tie = int(100000 - d1 * 1000 - d2 * 100 + order.p * 10 - min(order.et, 9999))
                candidates.append((score, tie, -shipper.id, -order.id, shipper.id, order.id))
        candidates.sort(reverse=True)
        assigned_shippers: Set[int] = set()
        assigned_orders: Set[int] = set()
        shipper_by_id = {s.id: s for s in free_shippers}
        for _, _, _, _, sid, oid in candidates:
            if sid in assigned_shippers or oid in assigned_orders:
                continue
            shipper = shipper_by_id.get(sid)
            order = orders.get(oid)
            if shipper is None or order is None or order.picked or order.delivered:
                continue
            assigned_shippers.add(sid)
            assigned_orders.add(oid)
            targets[sid] = (order.sx, order.sy)
            target_order[sid] = oid
            reserved_targets.add((order.sx, order.sy))
            self._order_claim[oid] = sid
            self._claim_age[oid] += 1
            if self._target_memory.get(sid) == oid:
                self._target_age[sid] += 1
            else:
                self._target_memory[sid] = oid
                self._target_age[sid] = 1
        for oid, sid in list(self._order_claim.items()):
            order = orders.get(oid)
            if order is None or order.delivered or order.picked:
                self._order_claim.pop(oid, None)
                self._claim_age.pop(oid, None)
            elif oid not in assigned_orders:
                self._order_claim.pop(oid, None)
                self._claim_age.pop(oid, None)
        for shipper in free_shippers:
            if shipper.id in targets:
                continue
            park = self._idle_parking_target(shipper, shippers, orders, reserved_targets)
            targets[shipper.id] = park
            target_order[shipper.id] = None
            reserved_targets.add(park)
            old_oid = self._target_memory.get(shipper.id)
            if old_oid is not None and self._order_claim.get(old_oid) == shipper.id:
                self._order_claim.pop(old_oid, None)
                self._claim_age.pop(old_oid, None)
            self._target_memory[shipper.id] = None
            self._target_age[shipper.id] = 0
        return (targets, target_order)

    def _constraint_tables(self, constraints: Tuple[Constraint, ...], agent: int):
        vertex: Dict[int, Set[Position]] = defaultdict(set)
        edge: Dict[int, Set[Tuple[Position, Position]]] = defaultdict(set)
        max_t = 0
        for c in constraints:
            if c.agent != agent:
                continue
            max_t = max(max_t, c.time)
            if c.kind == 'vertex':
                vertex[c.time].add(c.cell)
            elif c.kind == 'edge' and c.prev is not None:
                edge[c.time].add((c.prev, c.cell))
        return (vertex, edge, max_t)

    def _low_level_plan(self, agent: int, start: Position, goal: Position, constraints: Tuple[Constraint, ...], reserved_vertex: Optional[Dict[int, Set[Position]]]=None, reserved_edge: Optional[Dict[int, Set[Tuple[Position, Position]]]]=None, extra_avoid: Optional[Set[Position]]=None) -> Optional[List[Position]]:
        vertex_cons, edge_cons, max_constraint_t = self._constraint_tables(constraints, agent)
        if start == goal:
            has_future_vertex_block = any((start in cells for tm, cells in vertex_cons.items() if tm >= 1))
            if not has_future_vertex_block:
                return [start]
        if not is_valid_cell(goal, self.grid):
            return None
        dist = self._distance(start, goal)
        if dist >= INF:
            return None
        reserved_vertex = reserved_vertex or {}
        reserved_edge = reserved_edge or {}
        extra_avoid = extra_avoid or set()
        max_time = min(max(self.cbs_horizon + 2, dist + 4, max_constraint_t + 1), max(self.T, 1))

        def h(cell: Position) -> int:
            return self._distance(cell, goal)
        pq: List[Tuple[float, int, Position, int]] = []
        heapq.heappush(pq, (float(h(start)), 0, start, 0))
        best_g: Dict[Tuple[Position, int], float] = {(start, 0): 0.0}
        parent: Dict[Tuple[Position, int], Tuple[Optional[Tuple[Position, int]], Position]] = {(start, 0): (None, start)}
        counter = 0
        while pq:
            _, _, pos, tm = heapq.heappop(pq)
            g = best_g[pos, tm]
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
                if ng >= best_g.get(state, float('inf')):
                    continue
                best_g[state] = ng
                parent[state] = ((pos, tm), pos)
                counter += 1
                heapq.heappush(pq, (ng + h(nxt), counter, nxt, nt))
        return None

    @staticmethod
    def _path_at(path: List[Position], t: int) -> Position:
        if not path:
            raise ValueError('empty path')
        return path[t] if t < len(path) else path[-1]

    def _first_conflict(self, paths: Dict[int, List[Position]]) -> Optional[Tuple[str, int, int, int, Position, Optional[Position]]]:
        ids = sorted(paths)
        for tm in range(1, self.cbs_horizon + 1):
            occupied: Dict[Position, int] = {}
            for aid in ids:
                pos = self._path_at(paths[aid], tm)
                if pos in occupied:
                    return ('vertex', tm, occupied[pos], aid, pos, None)
                occupied[pos] = aid
            for i, a in enumerate(ids):
                a_prev, a_now = (self._path_at(paths[a], tm - 1), self._path_at(paths[a], tm))
                for b in ids[i + 1:]:
                    b_prev, b_now = (self._path_at(paths[b], tm - 1), self._path_at(paths[b], tm))
                    if a_prev == b_now and b_prev == a_now and (a_prev != a_now):
                        return ('edge', tm, a, b, a_now, a_prev)
        return None

    def _paths_cost(self, paths: Dict[int, List[Position]]) -> int:
        return sum((max(0, len(path) - 1) for path in paths.values()))

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
                if kind == 'vertex':
                    new_constraint = Constraint(agent=aid, time=tm, cell=cell, kind='vertex')
                else:
                    p0 = self._path_at(node.paths[aid], tm - 1)
                    p1 = self._path_at(node.paths[aid], tm)
                    new_constraint = Constraint(agent=aid, time=tm, cell=p1, kind='edge', prev=p0)
                new_constraints = node.constraints + (new_constraint,)
                new_paths = dict(node.paths)
                replanned = self._low_level_plan(aid, starts[aid], goals[aid], new_constraints)
                if replanned is None:
                    continue
                new_paths[aid] = replanned
                conflicts_left = 1 if self._first_conflict(new_paths) else 0
                heapq.heappush(open_list, CBSNode((conflicts_left, self._paths_cost(new_paths), expansions), new_constraints, new_paths))
        return self._prioritized_plan(starts, goals)

    def _prioritized_plan(self, starts: Dict[int, Position], goals: Dict[int, Position]) -> Dict[int, List[Position]]:
        paths: Dict[int, List[Position]] = {}
        reserved_vertex: Dict[int, Set[Position]] = defaultdict(set)
        reserved_edge: Dict[int, Set[Tuple[Position, Position]]] = defaultdict(set)

        def priority(aid: int) -> Tuple[int, int]:
            return (-min(self._stuck_count.get(aid, 0), 12), aid)
        for aid in sorted(starts, key=priority):
            start = starts[aid]
            goal = goals.get(aid, start)
            extra_avoid = {cell for cell, heat in self._cell_block_heat.items() if heat >= 5.0}
            if start in extra_avoid:
                extra_avoid.remove(start)
            path = self._low_level_plan(aid, start, goal, tuple(), reserved_vertex=reserved_vertex, reserved_edge=reserved_edge, extra_avoid=extra_avoid)
            if path is None:
                path = self._low_level_plan(aid, start, goal, tuple(), reserved_vertex=reserved_vertex, reserved_edge=reserved_edge)
            if path is None:
                path = [start]
            paths[aid] = path
            for tm in range(1, self.cbs_horizon + 2):
                pos = self._path_at(path, tm)
                prev = self._path_at(path, tm - 1)
                reserved_vertex[tm].add(pos)
                reserved_edge[tm].add((pos, prev))
        return paths

    def _safe_first_step_plan(self, starts: Dict[int, Position], goals: Dict[int, Position]) -> Dict[int, List[Position]]:
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
                candidates: List[Tuple[float, Position]] = [(3.0 + self._cell_penalty(start), start)]
                for _, nxt in self._neighbors(start, allow_wait=False):
                    if nxt in occupied or nxt in chosen.values():
                        continue
                    swap = any((nxt == starts[oid] and onext == start for oid, onext in chosen.items()))
                    if swap:
                        continue
                    candidates.append((self._cell_penalty(nxt) + self._distance(nxt, goals.get(aid, start)), nxt))
                desired = min(candidates, key=lambda x: x[0])[1]
            chosen[aid] = desired
            occupied.add(desired)
            fixed[aid] = [start, desired]
        return fixed

    def _has_delivery_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        return any((oid in orders and (not orders[oid].delivered) and ((orders[oid].ex, orders[oid].ey) == pos) for oid in shipper.bag))

    def _can_pick_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        waiting_here = self._waiting_by_pickup.get(pos)
        if waiting_here is None:
            waiting_here = [order for order in orders.values() if not order.picked and (not order.delivered) and ((order.sx, order.sy) == pos)]
        for order in waiting_here:
            if order.picked or order.delivered:
                continue
            claim_owner = self._order_claim.get(order.id)
            if claim_owner is not None and claim_owner != shipper.id:
                continue
            if self._can_carry_from_state(shipper, order, orders):
                return True
        return False

    def _fallback_delivery_action(self, shipper: Shipper, orders: Dict[int, Order], t: int) -> Optional[Action]:
        if not shipper.bag:
            return None
        if self._has_delivery_at(shipper, orders, shipper.position):
            return ('S', 2)
        target = self._delivery_target(shipper, orders, t)
        if target is None or target == shipper.position:
            return None
        move = self._next_move(shipper.position, target)
        nxt = valid_next_pos(shipper.position, move, self.grid)
        if nxt != shipper.position:
            return (move, 2 if self._has_delivery_at(shipper, orders, nxt) else 0)
        candidates: List[Tuple[float, Move, Position]] = []
        for mv, nb in self._neighbors(shipper.position, allow_wait=False):
            d = self._distance(nb, target)
            if d < INF:
                candidates.append((d + self._cell_penalty(nb), mv, nb))
        if not candidates:
            return None
        _, mv, nb = min(candidates, key=lambda x: x[0])
        return (mv, 2 if self._has_delivery_at(shipper, orders, nb) else 0)

    def _actions_from_paths(self, obs: dict, paths: Dict[int, List[Position]], target_order: Dict[int, Optional[int]]) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs['orders']
        shippers: List[Shipper] = obs['shippers']
        actions: Dict[int, Action] = {}
        for shipper in sorted(shippers, key=lambda s: s.id):
            path = paths.get(shipper.id, [shipper.position])
            next_pos = self._path_at(path, 1) if len(path) > 1 else shipper.position
            move = self._move_between(shipper.position, next_pos)
            if self._has_delivery_at(shipper, orders, next_pos):
                actions[shipper.id] = (move, 2)
                continue
            if shipper.bag and move == 'S' and (self._idle_count.get(shipper.id, 0) >= 20):
                fallback = self._fallback_delivery_action(shipper, orders, int(obs['t']))
                if fallback is not None:
                    actions[shipper.id] = fallback
                    continue
            oid = target_order.get(shipper.id)
            if oid is not None:
                order = orders.get(oid)
                if order and (not order.picked) and (not order.delivered) and ((order.sx, order.sy) == next_pos):
                    actions[shipper.id] = (move, 1)
                    continue
            if not shipper.bag and self._can_pick_at(shipper, orders, next_pos):
                actions[shipper.id] = (move, 1)
            else:
                actions[shipper.id] = (move, 0)
        return actions

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        self._refresh_traffic_state(obs)
        shippers: List[Shipper] = obs['shippers']
        targets, target_order = self._assign_targets(obs)
        starts = {shipper.id: shipper.position for shipper in shippers}
        if self.N <= 15:
            paths = self._cbs_plan(starts, targets)
        else:
            paths = self._safe_first_step_plan(starts, targets)
        actions = self._actions_from_paths(obs, paths, target_order)
        self._last_actions = actions
        self._last_desired = {sid: valid_next_pos(starts[sid], actions.get(sid, ('S', 0))[0], self.grid) for sid in starts}
        return actions

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()
        while not obs.get('done', False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)

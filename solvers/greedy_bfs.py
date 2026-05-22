# from __future__ import annotations

# import time
# from collections import deque, defaultdict
# from typing import Dict, Iterable, List, Optional, Tuple

# from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos, manhattan
# from solvers.solver import Solver

# Move = str
# Position = Tuple[int, int]
# Action = Tuple[Move, object]

# INF = 10**9
# MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


# class GreedyBFS(Solver):
#     """
#     Greedy BFS cải tiến cho Online MAPD:
#     - Heatmap theo dõi mật độ đơn (surge/hotspot)
#     - Heuristic đa tiêu chí (dist, priority, weight, urgency)
#     - Auction phân công đơn tránh trùng
#     - Reservation table giảm xung đột
#     - Relocate về tâm các đơn nếu rảnh
#     """

#     method_name = "GreedyBFS"

#     def __init__(self, env: DeliveryEnv):
#         super().__init__(env)
#         self._distance_cache: Dict[Tuple[Position, Position], int] = {}
#         self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        
#         # Heatmap
#         self.heatmap: Dict[Position, float] = {}
#         self.decay_rate = 0.95
#         for r in range(len(self.grid)):
#             for c in range(len(self.grid[0])):
#                 if self.grid[r][c] == 0:
#                     self.heatmap[(r, c)] = 0.0

#         # Precompute distances nếu grid nhỏ
#         self._precompute_all_distances()

#         # Reservation: (time, position) -> agent_id
#         self._reserved: Dict[Tuple[int, Position], int] = {}

#     # ------------------------------------------------------------------
#     # Precompute distances (nếu grid đủ nhỏ)
#     # ------------------------------------------------------------------
#     def _precompute_all_distances(self):
#         free = [(r, c) for r, row in enumerate(self.grid) for c, v in enumerate(row) if v == 0]
#         if len(free) > 400:
#             return
#         for p in free:
#             for q in free:
#                 self._distance_cache[(p, q)] = INF if p != q else 0
#         for start in free:
#             queue = deque([start])
#             dist = {start: 0}
#             while queue:
#                 r, c = queue.popleft()
#                 for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
#                     nr, nc = r+dr, c+dc
#                     if is_valid_cell((nr, nc), self.grid) and (nr, nc) not in dist:
#                         dist[(nr, nc)] = dist[(r, c)] + 1
#                         queue.append((nr, nc))
#             for goal, d in dist.items():
#                 self._distance_cache[(start, goal)] = d
#                 if self._distance_cache.get((goal, start), INF) > d:
#                     self._distance_cache[(goal, start)] = d

#     # ------------------------------------------------------------------
#     # BFS utilities
#     # ------------------------------------------------------------------
#     def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
#         for move in MOVES:
#             nxt = valid_next_pos(pos, move, self.grid)
#             if nxt != pos:
#                 yield move, nxt

#     def _bfs_parents(self, start: Position, goal: Position) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
#         if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
#             return None
#         queue = deque([start])
#         parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}
#         while queue:
#             current = queue.popleft()
#             if current == goal:
#                 return parent
#             for move, nxt in self._neighbors(current):
#                 if nxt in parent:
#                     continue
#                 parent[nxt] = (current, move)
#                 queue.append(nxt)
#         return None

#     def _distance(self, start: Position, goal: Position) -> int:
#         if start == goal:
#             return 0
#         key = (start, goal)
#         if key in self._distance_cache:
#             return self._distance_cache[key]
#         parent = self._bfs_parents(start, goal)
#         if parent is None or goal not in parent:
#             self._distance_cache[key] = INF
#             return INF
#         distance = 0
#         cur = goal
#         while cur != start:
#             prev, _ = parent[cur]
#             if prev is None:
#                 self._distance_cache[key] = INF
#                 return INF
#             cur = prev
#             distance += 1
#         self._distance_cache[key] = distance
#         return distance

#     def _next_move(self, start: Position, goal: Position, agent_id: int, current_time: int) -> Move:
#         """Tìm bước đi đầu tiên, tránh xung đột qua reservation."""
#         if start == goal:
#             return "S"
#         # Dùng cache nếu không cần tránh xung đột
#         if not self._reserved:
#             key = (start, goal)
#             if key in self._next_move_cache:
#                 return self._next_move_cache[key]
#         # BFS có xét reservation
#         queue = deque([start])
#         parent = {start: (None, "S")}
#         while queue:
#             cur = queue.popleft()
#             if cur == goal:
#                 break
#             for move, nxt in self._neighbors(cur):
#                 # Kiểm tra xung đột: ô nxt có bị chiếm tại thời điểm current_time+1 không
#                 if (current_time + 1, nxt) in self._reserved and self._reserved[(current_time+1, nxt)] != agent_id:
#                     continue
#                 if nxt not in parent:
#                     parent[nxt] = (cur, move)
#                     queue.append(nxt)
#         if goal not in parent:
#             return "S"
#         # Truy ngược
#         cur = goal
#         while True:
#             prev, move = parent[cur]
#             if prev is None:
#                 return "S"
#             if prev == start:
#                 if not self._reserved:
#                     self._next_move_cache[(start, goal)] = move
#                 return move
#             cur = prev

#     # ------------------------------------------------------------------
#     # Heuristic chọn đơn
#     # ------------------------------------------------------------------
#     def _score_pickup(self, shipper: Shipper, order: Order, current_t: int) -> float:
#         """Tính điểm ưu tiên nhặt (càng nhỏ càng tốt)."""
#         dist = self._distance(shipper.position, (order.sx, order.sy))
#         if dist >= INF:
#             return float('inf')
#         urgency = max(0, order.et - current_t)  # càng nhỏ càng gấp
#         # dist - 2*priority - weight/5 + urgency*0.5
#         return dist - 2 * order.p - (order.w / 5.0) + urgency * 0.5

#     def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[Order]:
#         carried = [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]
#         if not carried:
#             return None
#         return min(carried, key=lambda o: (
#             self._distance(shipper.position, (o.ex, o.ey)),
#             o.et,
#             -o.p,
#             o.id
#         ))

#     def _select_pickup(self, shipper: Shipper, orders: Dict[int, Order], current_t: int,
#                        reserved_ids: set[int]) -> Optional[Order]:
#         best = None
#         best_score = float('inf')
#         for o in orders.values():
#             if o.id in reserved_ids or o.picked or o.delivered:
#                 continue
#             if not shipper.can_carry(o, orders):
#                 continue
#             score = self._score_pickup(shipper, o, current_t)
#             if score < best_score:
#                 best_score = score
#                 best = o
#         return best

#     # ------------------------------------------------------------------
#     # Auction phân công đơn
#     # ------------------------------------------------------------------
#     def _auction_assignments(self, shippers: List[Shipper], orders: Dict[int, Order],
#                              current_t: int) -> Dict[int, List[int]]:
#         """Trả về dict: shipper_id -> list of order_id được phân công (tối đa 1)."""
#         pending = [o for o in orders.values() if not o.picked and not o.delivered]
#         if not pending:
#             return {}
#         scores = {}
#         for s in shippers:
#             for o in pending:
#                 if s.can_carry(o, orders):
#                     scores[(s.id, o.id)] = self._score_pickup(s, o, current_t)
#         assignment = defaultdict(list)
#         sorted_orders = sorted(pending, key=lambda o: (-o.p, o.et))
#         for o in sorted_orders:
#             best_shipper = None
#             best_score = float('inf')
#             for s in shippers:
#                 if s.id in assignment and len(assignment[s.id]) >= 1:
#                     continue
#                 score = scores.get((s.id, o.id), float('inf'))
#                 if score < best_score:
#                     best_score = score
#                     best_shipper = s.id
#             if best_shipper is not None:
#                 assignment[best_shipper].append(o.id)
#         return assignment

#     # ------------------------------------------------------------------
#     # Action helpers với reservation
#     # ------------------------------------------------------------------
#     def _delivery_action(self, shipper: Shipper, order: Order, current_t: int) -> Action:
#         goal = (order.ex, order.ey)
#         move = self._next_move(shipper.position, goal, shipper.id, current_t)
#         nxt = valid_next_pos(shipper.position, move, self.grid)
#         self._reserved[(current_t+1, nxt)] = shipper.id
#         return (move, 2) if nxt == goal else (move, 0)

#     def _pickup_action(self, shipper: Shipper, order: Order, current_t: int) -> Action:
#         goal = (order.sx, order.sy)
#         move = self._next_move(shipper.position, goal, shipper.id, current_t)
#         nxt = valid_next_pos(shipper.position, move, self.grid)
#         self._reserved[(current_t+1, nxt)] = shipper.id
#         return (move, 1) if nxt == goal else (move, 0)

#     def _relocate_action(self, shipper: Shipper, target: Position, current_t: int) -> Action:
#         if shipper.position == target:
#             return ("S", 0)
#         move = self._next_move(shipper.position, target, shipper.id, current_t)
#         nxt = valid_next_pos(shipper.position, move, self.grid)
#         self._reserved[(current_t+1, nxt)] = shipper.id
#         return (move, 0)

#     # ------------------------------------------------------------------
#     # Heatmap cập nhật và xác định tâm đơn
#     # ------------------------------------------------------------------
#     def _update_heatmap(self, new_order_ids: List[int], orders: Dict[int, Order]):
#         for oid in new_order_ids:
#             o = orders.get(oid)
#             if o:
#                 self.heatmap[(o.sx, o.sy)] += 1.0
#         for pos in self.heatmap:
#             self.heatmap[pos] *= self.decay_rate

#     def _get_center_of_orders(self, orders: Dict[int, Order]) -> Position:
#         if not orders:
#             return (len(self.grid)//2, len(self.grid[0])//2)
#         total_r = sum(o.sx for o in orders.values())
#         total_c = sum(o.sy for o in orders.values())
#         center = (total_r // len(orders), total_c // len(orders))
#         if is_valid_cell(center, self.grid):
#             return center
#         # Fallback: tìm ô trống gần nhất
#         best = center
#         best_dist = INF
#         for pos in self.heatmap:
#             d = manhattan(pos[0], pos[1], center[0], center[1])
#             if d < best_dist:
#                 best_dist = d
#                 best = pos
#         return best

#     # ------------------------------------------------------------------
#     # Quyết định actions tổng hợp
#     # ------------------------------------------------------------------
#     def _decide_actions(self, obs: dict) -> Dict[int, Action]:
#         current_t = obs["t"]
#         orders = obs["orders"]
#         shippers = obs["shippers"]
#         new_ids = obs.get("new_order_ids", [])

#         # Cập nhật heatmap
#         self._update_heatmap(new_ids, orders)

#         # Xóa reservation cũ
#         to_del = [k for k in self._reserved if k[0] <= current_t]
#         for k in to_del:
#             del self._reserved[k]

#         # Auction phân công đơn
#         assignments = self._auction_assignments(shippers, orders, current_t)

#         actions = {}
#         reserved_pickups = set()

#         for shipper in sorted(shippers, key=lambda s: s.id):
#             # Ưu tiên giao hàng
#             delivery = self._select_delivery(shipper, orders)
#             if delivery:
#                 actions[shipper.id] = self._delivery_action(shipper, delivery, current_t)
#                 continue

#             # Nếu được auction gán đơn
#             assigned = assignments.get(shipper.id, [])
#             if assigned:
#                 oid = assigned[0]
#                 order = orders.get(oid)
#                 if order and not order.picked and not order.delivered and shipper.can_carry(order, orders):
#                     reserved_pickups.add(oid)
#                     actions[shipper.id] = self._pickup_action(shipper, order, current_t)
#                     continue

#             # Chọn đơn heuristic
#             pickup = self._select_pickup(shipper, orders, current_t, reserved_pickups)
#             if pickup:
#                 reserved_pickups.add(pickup.id)
#                 actions[shipper.id] = self._pickup_action(shipper, pickup, current_t)
#                 continue

#             # Rảnh: relocate về tâm các đơn
#             target = self._get_center_of_orders(orders)
#             actions[shipper.id] = self._relocate_action(shipper, target, current_t)

#         return actions

#     # ------------------------------------------------------------------
#     # Main loop
#     # ------------------------------------------------------------------
#     def run(self) -> dict:
#         start_time = time.time()
#         obs = self.env.reset()
#         while not obs.get("done", False):
#             actions = self._decide_actions(obs)
#             obs, _, done, _ = self.env.step(actions)
#             if done:
#                 break
#         return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
    
# # python run_test.py --method GreedyBFS --config test_config.txt --out results


from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")

# Hằng số phần thưởng cơ bản (giống env)
def r_base(w: float) -> float:
    if w <= 0.2: return 4.0
    if w <= 3.0: return 10.0
    if w <= 10.0: return 15.0
    if w <= 30.0: return 20.0
    return 30.0


class GreedyBFS(Solver):
    """
    Greedy BFS kết hợp MARL tabular hoàn chỉnh:
    - V(s): Heatmap 2 lớp (Gaussian-like spreading)
    - Q(s,a): Phương trình Bellman + Urgency Bonus
    - Phối hợp đa tác tử: Tránh va chạm động (Repulsive Field & Reservation)
    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}

        # Khởi tạo Value function V(s)
        self.V: Dict[Position, float] = {}
        self.decay_rate = 0.95          
        for r in range(len(self.grid)):
            for c in range(len(self.grid[0])):
                if self.grid[r][c] == 0:
                    self.V[(r, c)] = 0.0

        # Precompute distances nếu grid nhỏ (để tối ưu tốc độ)
        self._precompute_all_distances()

        # Reservation: (time, position) -> agent_id
        self._reserved: Dict[Tuple[int, Position], int] = {}

    # ------------------------------------------------------------------
    # Precompute distances
    # ------------------------------------------------------------------
    def _precompute_all_distances(self):
        free = [(r, c) for r, row in enumerate(self.grid) for c, v in enumerate(row) if v == 0]
        if len(free) > 400:
            return
        for p in free:
            for q in free:
                self._distance_cache[(p, q)] = INF if p != q else 0
        for start in free:
            queue = deque([start])
            dist = {start: 0}
            while queue:
                r, c = queue.popleft()
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = r+dr, c+dc
                    if is_valid_cell((nr, nc), self.grid) and (nr, nc) not in dist:
                        dist[(nr, nc)] = dist[(r, c)] + 1
                        queue.append((nr, nc))
            for goal, d in dist.items():
                self._distance_cache[(start, goal)] = d
                if self._distance_cache.get((goal, start), INF) > d:
                    self._distance_cache[(goal, start)] = d

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(self, start: Position, goal: Position) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None
        queue = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}
        while queue:
            current = queue.popleft()
            if current == goal:
                return parent
            for move, nxt in self._neighbors(current):
                if nxt in parent:
                    continue
                parent[nxt] = (current, move)
                queue.append(nxt)
        return None

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal: return 0
        key = (start, goal)
        if key in self._distance_cache: return self._distance_cache[key]
        
        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._distance_cache[key] = INF
            return INF
            
        distance = 0
        cur = goal
        while cur != start:
            prev, _ = parent[cur]
            if prev is None:
                self._distance_cache[key] = INF
                return INF
            cur = prev
            distance += 1
            
        self._distance_cache[key] = distance
        return distance

    def _next_move(self, start: Position, goal: Position, agent_id: int, current_time: int) -> Move:
        """Tìm bước đi đầu tiên, tránh xung đột qua reservation."""
        if start == goal: return "S"
        
        if not self._reserved:
            key = (start, goal)
            if key in self._next_move_cache:
                return self._next_move_cache[key]
                
        queue = deque([start])
        parent = {start: (None, "S")}
        while queue:
            cur = queue.popleft()
            if cur == goal: break
            
            for move, nxt in self._neighbors(cur):
                if (current_time + 1, nxt) in self._reserved and self._reserved[(current_time+1, nxt)] != agent_id:
                    continue
                if nxt not in parent:
                    parent[nxt] = (cur, move)
                    queue.append(nxt)
                    
        if goal not in parent: return "S"
        
        cur = goal
        while True:
            prev, move = parent[cur]
            if prev is None: return "S"
            if prev == start:
                if not self._reserved:
                    self._next_move_cache[(start, goal)] = move
                return move
            cur = prev

    # ------------------------------------------------------------------
    # Value Function V(s) – Cập nhật heatmap 2 lớp lan truyền
    # ------------------------------------------------------------------
    def _update_V(self, new_order_ids: List[int], orders: Dict[int, Order]):
        """Cập nhật V(s) với độ phủ rộng hơn (Gaussian-like spreading)."""
        for oid in new_order_ids:
            o = orders.get(oid)
            if not o: continue
            pos = (o.sx, o.sy)
            
            reward = r_base(o.w) * {1: 1.0, 2: 2.0, 3: 3.0}[o.p]
            self.V[pos] += reward
            
            # Lớp 1: Bán kính 1 (Gamma = 0.6)
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                adj = (pos[0]+dr, pos[1]+dc)
                if adj in self.V:
                    self.V[adj] += reward * 0.6
                    
                    # Lớp 2: Bán kính 2 (Gamma = 0.2) tạo độ mượt
                    for ddr, ddc in [(0,1), (1,0), (0,-1), (-1,0)]:
                        adj2 = (adj[0]+ddr, adj[1]+ddc)
                        if adj2 in self.V and adj2 != pos:
                            self.V[adj2] += reward * 0.2

        # Decay (quên dần) để giữ tính dẻo (plasticity) cho mạng lưới
        for p in self.V:
            self.V[p] *= self.decay_rate

    # ------------------------------------------------------------------
    # Action-Value Q(s,a) cho pickup
    # ------------------------------------------------------------------
    def _q_pickup(self, shipper: Shipper, order: Order, current_t: int) -> float:
        """Tính Q(s,a) theo Bellman: Q = immediate_reward + gamma * V(destination)"""
        dist_to_pickup = self._distance(shipper.position, (order.sx, order.sy))
        if dist_to_pickup >= INF:
            return -1e9

        base = r_base(order.w)
        alpha = {1: 1.0, 2: 2.0, 3: 3.0}[order.p]
        expected_reward = base * alpha
        move_cost = dist_to_pickup * 0.01 
        immediate = expected_reward - move_cost

        dist_to_delivery = self._distance((order.sx, order.sy), (order.ex, order.ey))
        total_time = dist_to_pickup + dist_to_delivery
        time_margin = order.et - (current_t + total_time)

        if time_margin < 0:
            immediate *= 0.1  # Phạt nặng nếu rớt xuống khung BETA
        else:
            # Urgency Bonus: Càng sát giờ, Q-value càng bùng nổ
            urgency = (20.0 / (time_margin + 1.0)) * alpha
            immediate += urgency

        future = self.V.get((order.ex, order.ey), 0.0)
        gamma = 0.8
        return immediate + gamma * future

    def _select_pickup(self, shipper: Shipper, orders: Dict[int, Order],
                       reserved_ids: set[int], current_t: int) -> Optional[Order]:
        """Chọn đơn có Q(s,a) cao nhất."""
        best_order = None
        best_q = -1e9
        for o in orders.values():
            if o.id in reserved_ids or o.picked or o.delivered:
                continue
            if not shipper.can_carry(o, orders):
                continue
            q = self._q_pickup(shipper, o, current_t)
            if q > best_q:
                best_q = q
                best_order = o
        return best_order

    # ------------------------------------------------------------------
    # Giao hàng (Heuristic khoảng cách + deadline)
    # ------------------------------------------------------------------
    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[Order]:
        carried = [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]
        if not carried: return None
        return min(carried, key=lambda o: (
            self._distance(shipper.position, (o.ex, o.ey)),
            o.et,
            -o.p,
            o.id
        ))

    # ------------------------------------------------------------------
    # Action helpers với reservation
    # ------------------------------------------------------------------
    def _delivery_action(self, shipper: Shipper, order: Order, current_t: int) -> Action:
        goal = (order.ex, order.ey)
        move = self._next_move(shipper.position, goal, shipper.id, current_t)
        nxt = valid_next_pos(shipper.position, move, self.grid)
        self._reserved[(current_t+1, nxt)] = shipper.id
        return (move, 2) if nxt == goal else (move, 0)

    def _pickup_action(self, shipper: Shipper, order: Order, current_t: int) -> Action:
        goal = (order.sx, order.sy)
        move = self._next_move(shipper.position, goal, shipper.id, current_t)
        nxt = valid_next_pos(shipper.position, move, self.grid)
        self._reserved[(current_t+1, nxt)] = shipper.id
        return (move, 1) if nxt == goal else (move, 0)

    def _relocate_action(self, shipper: Shipper, target: Position, current_t: int) -> Action:
        if shipper.position == target: return ("S", 0)
        move = self._next_move(shipper.position, target, shipper.id, current_t)
        nxt = valid_next_pos(shipper.position, move, self.grid)
        self._reserved[(current_t+1, nxt)] = shipper.id
        return (move, 0)

    # ------------------------------------------------------------------
    # Quyết định actions với phối hợp đa tác tử (Repulsive Field)
    # ------------------------------------------------------------------
    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        current_t = obs["t"]
        orders = obs["orders"]
        shippers = obs["shippers"]
        new_ids = obs.get("new_order_ids", [])

        # 1. Cập nhật Value Function
        self._update_V(new_ids, orders)

        # 2. Xóa reservation cũ
        to_del = [k for k in self._reserved if k[0] <= current_t]
        for k in to_del:
            del self._reserved[k]

        # 3. Sắp xếp shipper: ưu tiên shipper rảnh (ít bag) và có V(position) cao
        sorted_shippers = sorted(shippers, key=lambda s: (len(s.bag), -self.V.get(s.position, 0.0)))

        actions = {}
        reserved_pickups = set()
        claimed_targets = []  # Lưu lại các điểm nóng đã bị chiếm để tạo trường đẩy

        for shipper in sorted_shippers:
            # Ưu tiên 1: Giao hàng nếu có
            delivery = self._select_delivery(shipper, orders)
            if delivery:
                actions[shipper.id] = self._delivery_action(shipper, delivery, current_t)
                continue

            # Ưu tiên 2: Chọn nhặt dựa trên Q-value
            pickup = self._select_pickup(shipper, orders, reserved_pickups, current_t)
            if pickup:
                reserved_pickups.add(pickup.id)
                actions[shipper.id] = self._pickup_action(shipper, pickup, current_t)
                continue

            # Ưu tiên 3: Rảnh -> Di chuyển về hotspot kết hợp trường đẩy (Repulsive Field)
            best_target = None
            best_val = -1e9
            
            for pos, val in self.V.items():
                if val <= 0.1: continue # Bỏ qua các ô quá nguội
                
                # Trừ điểm V(s) dựa trên các shipper đã quyết định đi về hướng này
                penalty = sum(15.0 / (1.0 + self._distance(pos, hp)) for hp in claimed_targets)
                adj_val = val - penalty
                
                if adj_val > best_val:
                    best_val = adj_val
                    best_target = pos

            # Fallback về giữa map nếu không có điểm nóng nào
            if not best_target:
                best_target = (len(self.grid)//2, len(self.grid[0])//2)
                
            claimed_targets.append(best_target)
            actions[shipper.id] = self._relocate_action(shipper, best_target, current_t)

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
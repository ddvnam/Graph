

from __future__ import annotations

"""
vrp_online.py — Online VRP Solver cho MAPD (Full cải tiến)
==========================================================

Cải tiến so với v1:
  ① Đừng bỏ đơn trễ hạn — tính đúng BETA reward thay vì return -INF
  ② Delivery-first: nếu có delivery trong ≤3 bước, ưu tiên giao ngay
  ③ Time-aware insertion: objective = maximize net_reward (không chỉ minimize distance)
  ④ Weight-aware assignment: tính move_cost theo w/W_max vào Hungarian matrix
  ⑤ Surge detection + pre-positioning khi phát hiện spike đơn
  ⑥ 2-opt local search sau mỗi insertion (ràng buộc pickup trước delivery)
  ⑦ Cluster pickup: gom đơn gần nhau (manhattan ≤ 3) assign cùng shipper
  ⑧ Anti-idle: shipper rảnh luôn nhận đơn dù xa

Kiến trúc cốt lõi (giữ từ v1):
  - Mỗi shipper duy trì ROUTE = list[Waypoint]
  - Cheapest Insertion + Hungarian assignment
  - Reservation table tránh va chạm
"""

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from env import (
    ALPHA, BETA,
    DeliveryEnv, Order, Shipper,
    delivery_reward, is_valid_cell, valid_next_pos, r_base,
)
from solvers.solver import Solver

Move     = str
Position = Tuple[int, int]
Action   = Tuple[Move, int]

INF   = 10 ** 9
MOVES = ("U", "D", "L", "R")

# Ngưỡng phát hiện surge: ≥ N đơn mới trong 1 bước
SURGE_THRESHOLD = 3
# Số bước nhìn trước để ưu tiên delivery-first
DELIVERY_FIRST_HORIZON = 3
# Bán kính cluster pickup
CLUSTER_RADIUS = 3


# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------
class Waypoint:
    __slots__ = ("pos", "order_id", "kind")

    def __init__(self, pos: Position, order_id: int, kind: str):
        self.pos      = pos
        self.order_id = order_id
        self.kind     = kind

    def op(self) -> int:
        return 1 if self.kind == "pickup" else 2

    def __repr__(self) -> str:
        return f"WP({self.kind[:1].upper()}#{self.order_id}@{self.pos})"


# ---------------------------------------------------------------------------
# VRPOrToolsSolver
# ---------------------------------------------------------------------------
class VRPOrToolsSolver(Solver):
    method_name = "VRP-OrTools"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)

        self._dist_cache: Dict[Tuple[Position, Position], int] = {}
        self._move_cache: Dict[Tuple[Position, Position], Move] = {}
        self._precompute_distances()

        self._routes: Dict[int, List[Waypoint]] = {}
        self._routed_orders: Set[int] = set()
        self._reserved: Dict[Tuple[int, Position], int] = {}

        # ⑤ Surge detection
        self._recent_new_counts: List[int] = []   # window 3 bước gần nhất
        self._surge_active: bool = False

    # ================================================================
    # BFS helpers
    # ================================================================

    def _precompute_distances(self):
        free = [(r, c) for r, row in enumerate(self.grid)
                for c, v in enumerate(row) if v == 0]
        if len(free) > 600:
            return
        for start in free:
            q = deque([start])
            dist: Dict[Position, int] = {start: 0}
            while q:
                r, c = q.popleft()
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nb = (r+dr, c+dc)
                    if is_valid_cell(nb, self.grid) and nb not in dist:
                        dist[nb] = dist[(r,c)] + 1
                        q.append(nb)
            for goal, d in dist.items():
                self._dist_cache[(start, goal)] = d

    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for m in MOVES:
            nxt = valid_next_pos(pos, m, self.grid)
            if nxt != pos:
                yield m, nxt

    def _distance(self, a: Position, b: Position) -> int:
        if a == b:
            return 0
        key = (a, b)
        if key in self._dist_cache:
            return self._dist_cache[key]
        q = deque([a])
        visited = {a: 0}
        while q:
            cur = q.popleft()
            if cur == b:
                break
            for _, nb in self._neighbors(cur):
                if nb not in visited:
                    visited[nb] = visited[cur] + 1
                    q.append(nb)
        d = visited.get(b, INF)
        self._dist_cache[key] = d
        return d

    def _next_move(self, start: Position, goal: Position,
                   agent_id: int, t: int) -> Move:
        if start == goal:
            return "S"
        if not self._reserved:
            key = (start, goal)
            if key in self._move_cache:
                return self._move_cache[key]

        q: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}
        found = False
        while q:
            cur = q.popleft()
            if cur == goal:
                found = True
                break
            for mv, nb in self._neighbors(cur):
                res_key = (t + 1, nb)
                if res_key in self._reserved and self._reserved[res_key] != agent_id:
                    continue
                if nb not in parent:
                    parent[nb] = (cur, mv)
                    q.append(nb)

        if not found or goal not in parent:
            return "S"

        cur = goal
        while True:
            prev, mv = parent[cur]
            if prev is None:
                return "S"
            if prev == start:
                if not self._reserved:
                    self._move_cache[(start, goal)] = mv
                return mv
            cur = prev

    # ================================================================
    # ① Reward estimation — tính đúng cả BETA (giao trễ)
    # ================================================================

    def _estimate_reward(self, order: Order, t_arrive_pickup: int,
                         dist_p_to_d: int, T: int) -> float:
        """
        Ước lượng reward thực tế — KHÔNG bỏ đơn trễ hạn.
        Dùng đúng công thức env: ALPHA nếu đúng hạn, BETA nếu trễ.
        """
        t_deliver = t_arrive_pickup + dist_p_to_d
        if t_deliver > T:
            return 0.0   # Hết thời gian sim, thực sự không giao được
        return delivery_reward(order, t_deliver, T)
        # delivery_reward đã xử lý cả on-time (ALPHA) lẫn late (BETA)

    def _max_possible_reward(self, order: Order) -> float:
        """Reward tối đa (giao ngay tại t=appear_t, dùng để normalize)."""
        rb = r_base(order.w)
        return ALPHA[order.p] * rb * 2.0  # upper bound

    # ================================================================
    # ④ Move cost ước tính theo weight
    # ================================================================

    def _est_move_cost(self, steps: int, w_carried: float, w_max: float) -> float:
        """Ước tính move cost cho `steps` bước mang `w_carried` kg."""
        return steps * 0.01 * (1.0 + w_carried / max(w_max, 1.0))

    # ================================================================
    # ③ + ④ Time-aware + weight-aware insertion score
    # ================================================================

    def _insertion_net_value(
        self,
        shipper: Shipper,
        order: Order,
        route: List[Waypoint],
        t: int,
        T: int,
    ) -> Tuple[float, float, List[Waypoint]]:
        """
        Tính net_value = expected_reward - delta_move_cost cho việc chèn order.
        Trả (net_value, delta_steps, new_route).
        net_value lớn hơn = tốt hơn (ngược với delta_cost nhỏ hơn).
        """
        pickup_pos   = (order.sx, order.sy)
        delivery_pos = (order.ex, order.ey)
        pw = Waypoint(pickup_pos,   order.id, "pickup")
        dw = Waypoint(delivery_pos, order.id, "delivery")

        n = len(route)
        positions = [shipper.position] + [wp.pos for wp in route]

        # Ước lượng w_carried hiện tại (đơn đang trong bag)
        w_carried_base = sum(
            getattr(order, 'w', 0)
            for wp in route if wp.kind == "delivery"
        ) if route else 0.0
        # Giới hạn thực tế
        w_carried_base = min(w_carried_base, shipper.W_max)

        best_net   = float("-inf")
        best_delta = float("inf")
        best_route: List[Waypoint] = []

        for i in range(n + 1):
            prev_p = positions[i]
            next_p = positions[i + 1] if i < n else None

            d_insert_p = self._distance(prev_p, pickup_pos)
            if next_p is not None:
                d_insert_p += self._distance(pickup_pos, next_p) - self._distance(prev_p, next_p)

            # ③ Ước tính t khi đến pickup
            # Steps để đi từ shipper.position đến positions[i] (theo route hiện tại)
            steps_to_i = sum(
                self._distance(positions[k], positions[k+1])
                for k in range(i)
            ) if i > 0 else 0
            t_at_pickup = t + steps_to_i + self._distance(positions[i], pickup_pos)

            for j in range(i, n + 1):
                positions_after_p = positions[:i+1] + [pickup_pos] + positions[i+1:]
                prev_d = positions_after_p[j + 1]
                next_d = positions_after_p[j + 2] if (j + 1) < len(positions_after_p) - 1 else None

                d_insert_d = self._distance(prev_d, delivery_pos)
                if next_d is not None:
                    d_insert_d += self._distance(delivery_pos, next_d) - self._distance(prev_d, next_d)

                delta = d_insert_p + d_insert_d

                # ③ Reward thực tế theo thời gian đến delivery
                steps_p_to_d = self._distance(pickup_pos, delivery_pos)
                exp_reward = self._estimate_reward(order, t_at_pickup, steps_p_to_d, T)

                if exp_reward <= 0:
                    continue  # Không giao được trong thời gian sim

                # ④ Move cost bổ sung do chèn thêm delta bước mang đơn nặng
                extra_move_cost = self._est_move_cost(delta, w_carried_base + order.w, shipper.W_max)

                net = exp_reward - extra_move_cost

                if net > best_net:
                    best_net   = net
                    best_delta = delta
                    new2 = list(route)
                    new2.insert(i, pw)
                    new2.insert(j + 1, dw)
                    best_route = new2

        return best_net, best_delta, best_route

    # ================================================================
    # ⑦ Cluster pickup — gom đơn gần nhau
    # ================================================================

    def _cluster_orders(self, orders: List[Order]) -> List[List[Order]]:
        """
        Gom các đơn có pickup gần nhau (manhattan ≤ CLUSTER_RADIUS) thành cluster.
        Dùng greedy: đơn đầu mỗi cluster là seed, các đơn tiếp theo join nếu đủ gần.
        """
        if not orders:
            return []
        remaining = list(orders)
        clusters: List[List[Order]] = []
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            new_remaining = []
            for o in remaining:
                dist = abs(o.sx - seed.sx) + abs(o.sy - seed.sy)
                if dist <= CLUSTER_RADIUS:
                    cluster.append(o)
                else:
                    new_remaining.append(o)
            remaining = new_remaining
            clusters.append(cluster)
        return clusters

    # ================================================================
    # Hungarian assignment — tích hợp ①②③④⑦
    # ================================================================

    def _assign_new_orders(
        self,
        shippers: List[Shipper],
        orders: Dict[int, Order],
        new_ids: List[int],
        t: int,
    ):
        pending = [
            orders[oid] for oid in new_ids
            if oid in orders
            and oid not in self._routed_orders
            and not orders[oid].picked
            and not orders[oid].delivered
        ]
        if not pending:
            return

        T = self.env.T
        n_s = len(shippers)

        # ⑦ Cluster đơn trước khi assign
        clusters = self._cluster_orders(pending)

        for cluster in clusters:
            # Build cost matrix cho cluster này
            n_o = len(cluster)
            # cost = -net_value (Hungarian minimize cost)
            cost_matrix = np.full((n_s, n_o), fill_value=1e9)
            insert_cache: Dict[Tuple[int, int], Tuple[float, List[Waypoint]]] = {}

            for i, s in enumerate(shippers):
                route = self._routes.get(s.id, [])
                for j, o in enumerate(cluster):
                    if not s.can_carry(o, orders):
                        continue
                    net_val, delta, new_route = self._insertion_net_value(s, o, route, t, T)
                    if net_val <= 0:
                        continue
                    cost_matrix[i, j] = -net_val  # minimize negative = maximize positive
                    insert_cache[(i, j)] = (delta, new_route)

            if cost_matrix.min() >= 1e8:
                # Không có assignment khả thi, thử anti-idle fallback
                self._force_assign_cluster(cluster, shippers, orders, t, T)
                continue

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_in_round: Set[int] = set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] >= 1e8:
                    continue
                s = shippers[i]
                o = cluster[j]
                if o.id in self._routed_orders or o.id in assigned_in_round:
                    continue
                _, new_route = insert_cache[(i, j)]
                self._routes[s.id] = new_route
                self._routed_orders.add(o.id)
                assigned_in_round.add(o.id)

            # Đơn còn lại trong cluster chưa được assign
            unassigned = [o for o in cluster if o.id not in self._routed_orders]
            if unassigned:
                self._force_assign_cluster(unassigned, shippers, orders, t, T)

    def _force_assign_cluster(
        self,
        orders_list: List[Order],
        shippers: List[Shipper],
        orders: Dict[int, Order],
        t: int,
        T: int,
    ):
        """⑧ Anti-idle: gán đơn cho shipper route ngắn nhất dù xa."""
        shippers_sorted = sorted(shippers, key=lambda s: len(self._routes.get(s.id, [])))
        for o in sorted(orders_list, key=lambda x: (-x.p, x.et)):
            if o.id in self._routed_orders:
                continue
            for s in shippers_sorted:
                if not s.can_carry(o, orders):
                    continue
                route = self._routes.get(s.id, [])
                _, _, new_route = self._insertion_net_value(s, o, route, t, T)
                if new_route:
                    self._routes[s.id] = new_route
                    self._routed_orders.add(o.id)
                    break

    # ================================================================
    # Replan cho đơn chưa routed
    # ================================================================

    def _replan_unrouted(
        self,
        shippers: List[Shipper],
        orders: Dict[int, Order],
        t: int,
    ):
        unrouted = [
            o for o in orders.values()
            if not o.picked and not o.delivered and o.id not in self._routed_orders
        ]
        if not unrouted:
            return

        T = self.env.T
        idle = [s for s in shippers if not self._routes.get(s.id)]
        candidates = idle if idle else sorted(
            shippers, key=lambda s: len(self._routes.get(s.id, []))
        )[:max(1, len(shippers) // 2)]

        for o in sorted(unrouted, key=lambda x: (-x.p, x.et)):
            best_s, best_route, best_val = None, None, float("-inf")
            for s in candidates:
                if not s.can_carry(o, orders):
                    continue
                route = self._routes.get(s.id, [])
                net_val, _, new_route = self._insertion_net_value(s, o, route, t, T)
                # ⑧ Anti-idle: chấp nhận cả net_val âm nhỏ nếu không có lựa chọn tốt hơn
                if net_val > best_val and new_route:
                    best_val, best_s, best_route = net_val, s, new_route
            if best_s is not None:
                self._routes[best_s.id] = best_route
                self._routed_orders.add(o.id)

    # ================================================================
    # ⑥ 2-opt local search trên route
    # ================================================================

    def _two_opt(self, route: List[Waypoint], start_pos: Position) -> List[Waypoint]:
        """
        2-opt cải thiện route: swap đoạn [i:j] nếu giảm tổng distance.
        Ràng buộc: với mỗi order_id, pickup phải đứng trước delivery.
        """
        if len(route) < 4:
            return route

        def total_dist(r: List[Waypoint]) -> int:
            cur = start_pos
            total = 0
            for wp in r:
                total += self._distance(cur, wp.pos)
                cur = wp.pos
            return total

        def pickup_before_delivery(r: List[Waypoint]) -> bool:
            seen: Set[int] = set()
            for wp in r:
                if wp.kind == "delivery" and wp.order_id not in seen:
                    return False
                if wp.kind == "pickup":
                    seen.add(wp.order_id)
            return True

        improved = True
        best = list(route)
        best_d = total_dist(best)

        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 2, len(best)):
                    # Reverse đoạn [i+1 .. j]
                    candidate = best[:i+1] + list(reversed(best[i+1:j+1])) + best[j+1:]
                    if not pickup_before_delivery(candidate):
                        continue
                    d = total_dist(candidate)
                    if d < best_d - 1:   # cải thiện ít nhất 1 bước
                        best = candidate
                        best_d = d
                        improved = True
                        break
                if improved:
                    break

        return best

    # ================================================================
    # ② Delivery-first: kiểm tra và đẩy delivery gần lên đầu
    # ================================================================

    def _apply_delivery_first(
        self,
        shipper: Shipper,
        route: List[Waypoint],
        orders: Dict[int, Order],
    ) -> List[Waypoint]:
        """
        Nếu có waypoint delivery trong DELIVERY_FIRST_HORIZON bước tới,
        đẩy nó lên đầu route (trước bất kỳ pickup nào chưa bắt đầu).
        """
        if not route:
            return route

        # Tìm delivery gần nhất trong horizon
        cur = shipper.position
        steps = 0
        for idx, wp in enumerate(route):
            steps += self._distance(cur, wp.pos)
            cur = wp.pos
            if steps > DELIVERY_FIRST_HORIZON:
                break
            if wp.kind == "delivery":
                o = orders.get(wp.order_id)
                if o and not o.delivered and o.id in shipper.bag:
                    # Đẩy delivery này lên đầu nếu nó không phải đã ở đầu
                    if idx > 0:
                        new_route = [route[idx]] + route[:idx] + route[idx+1:]
                        # Validate: pickup trước delivery
                        seen: Set[int] = set()
                        valid = True
                        for w in new_route:
                            if w.kind == "delivery" and w.order_id not in seen:
                                valid = False
                                break
                            if w.kind == "pickup":
                                seen.add(w.order_id)
                        if valid:
                            return new_route
        return route

    # ================================================================
    # ⑤ Surge detection
    # ================================================================

    def _update_surge(self, new_count: int) -> bool:
        self._recent_new_counts.append(new_count)
        if len(self._recent_new_counts) > 3:
            self._recent_new_counts.pop(0)
        avg = sum(self._recent_new_counts) / len(self._recent_new_counts)
        self._surge_active = avg >= SURGE_THRESHOLD
        return self._surge_active

    def _surge_relocate(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
    ) -> Optional[Position]:
        """Trả vị trí centroid đơn mới nhất để relocate trong surge."""
        recent = [
            o for o in orders.values()
            if not o.picked and not o.delivered
            and (t - o.appear_t) <= 5
        ]
        if not recent:
            return None
        cr = sum(o.sx for o in recent) // len(recent)
        cc = sum(o.sy for o in recent) // len(recent)
        target = (cr, cc)
        if is_valid_cell(target, self.grid):
            return target
        # Snap về ô trống gần nhất
        free = [(r, c) for r, row in enumerate(self.grid)
                for c, v in enumerate(row) if v == 0]
        if free:
            return min(free, key=lambda p: abs(p[0]-cr) + abs(p[1]-cc))
        return None

    # ================================================================
    # Route cleanup
    # ================================================================

    def _clean_routes(self, orders: Dict[int, Order]):
        for sid, route in self._routes.items():
            cleaned = []
            for wp in route:
                o = orders.get(wp.order_id)
                if o is None or o.delivered:
                    continue
                if wp.kind == "pickup" and o.picked:
                    continue
                cleaned.append(wp)
            self._routes[sid] = cleaned

    # ================================================================
    # Action từ route
    # ================================================================

    def _action_from_route(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
    ) -> Action:
        route = self._routes.get(shipper.id, [])

        # Bỏ waypoint đầu không còn hợp lệ
        while route:
            wp = route[0]
            o  = orders.get(wp.order_id)
            if o is None or o.delivered:
                route.pop(0); continue
            if wp.kind == "pickup" and o.picked:
                route.pop(0); continue
            break

        if not route:
            # ⑧ Anti-idle + ⑤ Surge: nếu có surge, relocate về centroid đơn mới
            if self._surge_active:
                target = self._surge_relocate(shipper, orders, t)
                if target and target != shipper.position:
                    move = self._next_move(shipper.position, target, shipper.id, t)
                    nxt  = valid_next_pos(shipper.position, move, self.grid)
                    self._reserved[(t + 1, nxt)] = shipper.id
                    return (move, 0)
            return ("S", 0)

        # ② Delivery-first
        route = self._apply_delivery_first(shipper, route, orders)
        self._routes[shipper.id] = route

        wp   = route[0]
        move = self._next_move(shipper.position, wp.pos, shipper.id, t)
        nxt  = valid_next_pos(shipper.position, move, self.grid)
        self._reserved[(t + 1, nxt)] = shipper.id

        op = 0
        if nxt == wp.pos:
            op = wp.op()
            route.pop(0)

        return (move, op)

    # ================================================================
    # Main step
    # ================================================================

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        t: int                   = obs["t"]
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper]  = obs["shippers"]
        new_ids: List[int]       = obs.get("new_order_ids", [])

        # Xóa reservation cũ
        stale = [k for k in self._reserved if k[0] <= t]
        for k in stale:
            del self._reserved[k]

        # ⑤ Cập nhật surge
        self._update_surge(len(new_ids))

        # Làm sạch route
        self._clean_routes(orders)

        # Assign đơn mới (⑦ cluster + ① BETA-aware + ③ time-aware + ④ weight-aware)
        if new_ids:
            self._assign_new_orders(shippers, orders, new_ids, t)

        # Replan đơn sót (⑧ anti-idle)
        self._replan_unrouted(shippers, orders, t)

        # ⑥ 2-opt sau mỗi vài bước (không chạy mỗi bước để tiết kiệm compute)
        if t % 5 == 0:
            for s in shippers:
                route = self._routes.get(s.id, [])
                if len(route) >= 4:
                    self._routes[s.id] = self._two_opt(route, s.position)

        # Tạo action
        actions: Dict[int, Action] = {}
        for s in sorted(shippers, key=lambda x: x.id):
            actions[s.id] = self._action_from_route(s, orders, t)

        return actions

    # ================================================================
    # Run
    # ================================================================

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        for s in obs["shippers"]:
            self._routes[s.id] = []

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )


#python run_test.py --method VRPOrToolsSolver --config test_config.txt --out results
"""
custom_solver.py — Heuristic solver for Online MAPD.

Thuật toán (xem heuristic.txt để biết chi tiết):
  1. BFS maps   : Khi đơn mới xuất hiện, chạy BFS từ điểm nhận / giao
                  đến mọi ô trên map. Xóa khi đơn được giao xong.
  2. Phân công  : Brute-force n^k cách gán k đơn mới cho n shipper.
                  Mỗi cách gán, thử mọi vị trí chèn hợp lệ (pickup, delivery)
                  vào chuỗi waypoint hiện tại của shipper.
                  Chọn cách tối đa hóa tổng phần thưởng ước lượng.
  3. Di chuyển  : Mỗi shipper chọn ô kề có khoảng cách BFS ngắn nhất
                  đến waypoint đầu tiên trong chuỗi. Bằng nhau → chọn ngẫu nhiên.
  4. Xung đột   : Phát hiện xung đột (cùng ô, hoán đổi), xác định các shipper
                  liên quan, liệt kê 5^n tổ hợp di chuyển, chọn tổ hợp tối thiểu
                  hóa phần thưởng bị mất. Bằng nhau → chọn ngẫu nhiên.
"""

from __future__ import annotations

import random
import time
from collections import Counter, deque
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

from env import (
    DeliveryEnv, Order, Shipper,
    delivery_reward, is_valid_cell, move_cost, valid_next_pos,
)
from solvers.solver import Solver

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Position = Tuple[int, int]
Waypoint = Tuple[int, bool]          # (order_id, is_pickup)
DistMap  = Dict[Position, int]

INF       = 10 ** 9
ALL_MOVES = ("S", "U", "D", "L", "R")

# ---------------------------------------------------------------------------
# Brute-force cap: nếu n^k > CAP thì dùng greedy để tránh timeout
# ---------------------------------------------------------------------------
BRUTE_CAP = 3000


class CustomSolver(Solver):
    method_name = "CustomSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)

        # dist_maps[order_id]['pickup' | 'delivery'][pos] = BFS distance
        self.dist_maps: Dict[int, Dict[str, DistMap]] = {}

        # sequences[shipper_id] = danh sách waypoint theo thứ tự thực hiện
        self.sequences: Dict[int, List[Waypoint]] = {}

        # Đơn chưa được phân công (tất cả shipper đầy hoặc không thể chở)
        self.unassigned: Set[int] = set()

        self._rng = random.Random(42)

    # ================================================================
    # BFS helpers
    # ================================================================

    def _bfs_from(self, source: Position) -> DistMap:
        """
        BFS từ source trên grid, trả dist[cell] = khoảng cách từ cell đến source.
        Vì grid vô hướng, dist(A→B) == dist(B→A), nên đây cũng là BFS "đến source".
        """
        grid = self.grid
        dist: DistMap = {source: 0}
        q: deque[Position] = deque([source])
        while q:
            pos = q.popleft()
            d = dist[pos]
            for m in ("U", "D", "L", "R"):
                nxt = valid_next_pos(pos, m, grid)
                if nxt != pos and nxt not in dist:
                    dist[nxt] = d + 1
                    q.append(nxt)
        return dist

    def _build_dist_maps(self, order: Order) -> None:
        """Tính BFS maps cho pickup và delivery của order, lưu vào self.dist_maps."""
        self.dist_maps[order.id] = {
            "pickup":   self._bfs_from((order.sx, order.sy)),
            "delivery": self._bfs_from((order.ex, order.ey)),
        }

    def _dist_to(self, order_id: int, is_pickup: bool, pos: Position) -> int:
        key = "pickup" if is_pickup else "delivery"
        return self.dist_maps.get(order_id, {}).get(key, {}).get(pos, INF)

    # ================================================================
    # Reward estimation
    # ================================================================

    def _estimate_reward(
        self,
        start_pos: Position,
        start_bag:  List[int],
        sequence:   List[Waypoint],
        orders:     Dict[int, Order],
        shipper:    Shipper,
        t_now:      int,
    ) -> float:
        """
        Ước lượng tổng phần thưởng tương lai của shipper khi theo sequence.
        Coi như chỉ mình shipper đó trên map (không tính xung đột).
        """
        pos  = start_pos
        bag  = list(start_bag)
        t    = t_now
        T    = self.env.T
        total = 0.0

        for oid, is_pickup in sequence:
            order = orders.get(oid)
            if order is None:
                continue
            d = self._dist_to(oid, is_pickup, pos)
            if d >= INF:
                continue

            # Chi phí di chuyển (w_carried không đổi trong đoạn này)
            w_carried = sum(orders[bid].w for bid in bag if bid in orders)
            total += d * move_cost(w_carried, shipper.W_max)

            t   += d
            pos  = (order.sx, order.sy) if is_pickup else (order.ex, order.ey)

            if is_pickup:
                bag.append(oid)
            else:
                if oid in bag:
                    bag.remove(oid)
                total += delivery_reward(order, t, T)

        return total

    def _is_feasible(
        self,
        start_bag: List[int],
        sequence:  List[Waypoint],
        orders:    Dict[int, Order],
        shipper:   Shipper,
    ) -> bool:
        """True nếu sequence không vi phạm K_max / W_max tại bất kỳ bước pickup nào."""
        bag = list(start_bag)
        for oid, is_pickup in sequence:
            if not is_pickup:
                if oid in bag:
                    bag.remove(oid)
                continue
            order = orders.get(oid)
            if order is None:
                continue
            w_carried = sum(orders[bid].w for bid in bag if bid in orders)
            if len(bag) >= shipper.K_max or w_carried + order.w > shipper.W_max:
                return False
            bag.append(oid)
        return True

    # ================================================================
    # Sequence insertion
    # ================================================================

    def _best_insert(
        self,
        sequence:  List[Waypoint],
        order:     Order,
        start_bag: List[int],
        orders:    Dict[int, Order],
        shipper:   Shipper,
        pos:       Position,
        t_now:     int,
    ) -> Optional[Tuple[float, List[Waypoint]]]:
        """
        Thử tất cả vị trí chèn hợp lệ (p_idx, d_idx) cho
        (pickup_waypoint, delivery_waypoint) của order vào sequence.

        Với sequence có k waypoint:
          - pickup  tại p_idx ∈ [0, k]
          - delivery tại d_idx ∈ [p_idx+1, k+1]  (sau khi đã chèn pickup)
        Tổng: (k+1)(k+2)/2 tổ hợp.

        Trả (best_reward, best_sequence) hoặc None nếu không có insertion hợp lệ.
        """
        k = len(sequence)
        best_rew: float = -float("inf")
        best_seqs: List[List[Waypoint]] = []

        for p in range(k + 1):
            temp = sequence[:p] + [(order.id, True)] + sequence[p:]
            for d in range(p + 1, k + 2):
                new_seq = temp[:d] + [(order.id, False)] + temp[d:]

                if not self._is_feasible(start_bag, new_seq, orders, shipper):
                    continue

                rew = self._estimate_reward(pos, start_bag, new_seq, orders, shipper, t_now)

                if rew > best_rew:
                    best_rew  = rew
                    best_seqs = [new_seq]
                elif rew == best_rew:
                    best_seqs.append(new_seq)

        if not best_seqs:
            return None

        return best_rew, self._rng.choice(best_seqs)

    # ================================================================
    # Assignment
    # ================================================================

    def _assign_new_orders(
        self,
        new_ids:  List[int],
        shippers: List[Shipper],
        orders:   Dict[int, Order],
        t:        int,
    ) -> None:
        """
        Brute-force n^k phân công; mỗi phân công thử chèn tối ưu.
        Nếu n^k > BRUTE_CAP, dùng greedy từng đơn.
        """
        if not new_ids:
            return

        n, k = len(shippers), len(new_ids)

        if n ** k > BRUTE_CAP:
            # Greedy: phân công từng đơn một
            for oid in new_ids:
                if not self._greedy_assign_one(oid, shippers, orders, t):
                    self.unassigned.add(oid)
            return

        # ---- Brute-force ----
        best_total: float        = -float("inf")
        best_seqs:  Optional[Dict[int, List[Waypoint]]] = None
        best_tied:  List[Dict[int, List[Waypoint]]]     = []

        for assign_tuple in product(range(n), repeat=k):
            # assign_tuple[i] = chỉ số shipper trong danh sách shippers cho new_ids[i]
            shipper_new: Dict[int, List[int]] = {s.id: [] for s in shippers}
            for i, s_idx in enumerate(assign_tuple):
                shipper_new[shippers[s_idx].id].append(new_ids[i])

            candidate: Dict[int, List[Waypoint]] = {}
            valid = True

            for s in shippers:
                seq = list(self.sequences.get(s.id, []))
                for oid in shipper_new[s.id]:
                    order = orders.get(oid)
                    if order is None:
                        valid = False   # đơn không tồn tại → combination này vô hiệu
                        break
                    result = self._best_insert(
                        seq, order, list(s.bag), orders, s, s.position, t
                    )
                    if result is None:
                        valid = False
                        break
                    _, seq = result
                if not valid:
                    break
                candidate[s.id] = seq

            if not valid:
                continue

            total = sum(
                self._estimate_reward(s.position, s.bag, candidate[s.id], orders, s, t)
                for s in shippers
            )

            if total > best_total:
                best_total = total
                best_tied  = [candidate]
            elif total == best_total:
                best_tied.append(candidate)

        if best_tied:
            self.sequences.update(self._rng.choice(best_tied))
        else:
            # Không tìm được phân công hợp lệ qua brute-force; fallback greedy
            for oid in new_ids:
                if not self._greedy_assign_one(oid, shippers, orders, t):
                    self.unassigned.add(oid)

    def _greedy_assign_one(
        self,
        oid:      int,
        shippers: List[Shipper],
        orders:   Dict[int, Order],
        t:        int,
    ) -> bool:
        """
        Gán một đơn oid cho shipper có lợi nhất (delta reward cao nhất).
        Trả True nếu gán thành công.
        """
        order = orders.get(oid)
        if order is None:
            return False

        best_delta = -float("inf")
        best_sid:   Optional[int]          = None
        best_seq:   Optional[List[Waypoint]] = None
        best_tied_sids: List[Tuple[int, List[Waypoint]]] = []

        for s in shippers:
            seq    = list(self.sequences.get(s.id, []))
            before = self._estimate_reward(s.position, s.bag, seq, orders, s, t)
            result = self._best_insert(seq, order, list(s.bag), orders, s, s.position, t)
            if result is None:
                continue
            after_rew, after_seq = result
            delta = after_rew - before

            if delta > best_delta:
                best_delta = delta
                best_tied_sids = [(s.id, after_seq)]
            elif delta == best_delta:
                best_tied_sids.append((s.id, after_seq))

        if not best_tied_sids:
            return False

        best_sid, best_seq = self._rng.choice(best_tied_sids)
        self.sequences[best_sid] = best_seq
        return True

    def _retry_unassigned(
        self,
        shippers: List[Shipper],
        orders:   Dict[int, Order],
        t:        int,
    ) -> None:
        """Thử lại phân công các đơn chưa được gán (shipper có thể đã rảnh chỗ)."""
        still_unassigned: Set[int] = set()
        for oid in list(self.unassigned):
            if oid not in orders:
                continue  # đơn đã giao hoặc hết hạn → bỏ
            if not self._greedy_assign_one(oid, shippers, orders, t):
                still_unassigned.add(oid)
        self.unassigned = still_unassigned

    # ================================================================
    # Desired moves
    # ================================================================

    def _desired_move(
        self,
        shipper: Shipper,
        orders:  Dict[int, Order],
    ) -> str:
        """
        Bước đi tối ưu đến waypoint đầu tiên trong chuỗi của shipper.
        Chọn ô kề có khoảng cách BFS nhỏ nhất đến goal.
        Nếu nhiều ô bằng nhau → chọn ngẫu nhiên.
        Nếu không có nhiệm vụ → đứng yên.
        """
        seq = self.sequences.get(shipper.id, [])
        if not seq:
            return "S"

        oid, is_pickup = seq[0]
        order = orders.get(oid)
        if order is None:
            return "S"

        goal = (order.sx, order.sy) if is_pickup else (order.ex, order.ey)
        if shipper.position == goal:
            return "S"

        cur_d = self._dist_to(oid, is_pickup, shipper.position)
        if cur_d >= INF:
            return "S"     # không thể đến đích

        min_d      = cur_d
        best_moves: List[str] = []

        for m in ("U", "D", "L", "R"):
            nxt = valid_next_pos(shipper.position, m, self.grid)
            if nxt == shipper.position:
                continue   # tường chặn
            d = self._dist_to(oid, is_pickup, nxt)
            if d < min_d:
                min_d      = d
                best_moves = [m]
            elif d == min_d:
                best_moves.append(m)

        return self._rng.choice(best_moves) if best_moves else "S"

    # ================================================================
    # Conflict detection & resolution
    # ================================================================

    def _next_positions(
        self,
        shippers: List[Shipper],
        moves:    Dict[int, str],
    ) -> Dict[int, Position]:
        return {
            s.id: valid_next_pos(s.position, moves.get(s.id, "S"), self.grid)
            for s in shippers
        }

    def _has_conflict(
        self,
        shippers:       List[Shipper],
        next_pos_map:   Dict[int, Position],
    ) -> bool:
        """True nếu có xung đột cùng ô hoặc hoán đổi vị trí."""
        cells = list(next_pos_map.values())
        if len(cells) != len(set(cells)):
            return True     # hai shipper cùng muốn đến một ô

        # Hoán đổi: A đang ở X muốn đến Y, B đang ở Y muốn đến X
        pos_to_sid = {s.position: s.id for s in shippers}
        for s in shippers:
            nxt = next_pos_map[s.id]
            if nxt == s.position:
                continue
            other_sid = pos_to_sid.get(nxt)
            if other_sid is not None and next_pos_map.get(other_sid) == s.position:
                return True

        return False

    def _find_involved(
        self,
        shippers:     List[Shipper],
        next_pos_map: Dict[int, Position],
    ) -> List[Shipper]:
        """
        Trả danh sách shipper liên quan đến xung đột:
        - Trực tiếp: muốn đi vào ô đã có shipper khác, hoặc tham gia hoán đổi.
        - Gián tiếp: muốn đi vào ô kề với shipper trực tiếp xung đột.
        """
        # Tìm shipper trực tiếp xung đột
        cell_count  = Counter(next_pos_map.values())
        conflicted: Set[int] = set()

        for s in shippers:
            if cell_count[next_pos_map[s.id]] > 1:
                conflicted.add(s.id)

        pos_to_sid = {s.position: s.id for s in shippers}
        for s in shippers:
            nxt = next_pos_map[s.id]
            if nxt != s.position:
                other = pos_to_sid.get(nxt)
                if other is not None and next_pos_map.get(other) == s.position:
                    conflicted.add(s.id)
                    conflicted.add(other)

        # Mở rộng sang shipper gián tiếp (muốn đi vào ô kề với shipper xung đột)
        conflicted_positions = {s.position for s in shippers if s.id in conflicted}
        for s in shippers:
            if s.id in conflicted:
                continue
            nxt = next_pos_map[s.id]
            for m in ("U", "D", "L", "R"):
                adj = valid_next_pos(nxt, m, self.grid)
                if adj in conflicted_positions:
                    conflicted.add(s.id)
                    break

        return [s for s in shippers if s.id in conflicted]

    def _move_score(
        self,
        involved:     List[Shipper],
        candidate:    Dict[int, str],
        next_pos_map: Dict[int, Position],
        orders:       Dict[int, Order],
        t:            int,
    ) -> float:
        """
        Ước lượng tổng phần thưởng tương lai cho các shipper liên quan
        với bộ di chuyển candidate. Dùng để chọn bộ di chuyển tốt nhất.
        """
        total = 0.0
        for s in involved:
            new_pos = next_pos_map[s.id]
            seq     = self.sequences.get(s.id, [])
            total  += self._estimate_reward(new_pos, s.bag, seq, orders, s, t + 1)
        return total

    def _resolve_conflicts(
        self,
        shippers: List[Shipper],
        desired:  Dict[int, str],
        orders:   Dict[int, Order],
        t:        int,
    ) -> Dict[int, str]:
        """
        Phát hiện và giải quyết xung đột bằng cách liệt kê 5^n tổ hợp
        di chuyển cho n shipper liên quan.
        Lặp tối đa 5 vòng để xử lý xung đột dây chuyền.
        """
        moves = dict(desired)

        for _ in range(5):
            nxt_map = self._next_positions(shippers, moves)
            if not self._has_conflict(shippers, nxt_map):
                break

            involved = self._find_involved(shippers, nxt_map)
            if not involved:
                break

            best_score = -float("inf")
            best_combos: List[Tuple] = []

            for combo in product(ALL_MOVES, repeat=len(involved)):
                # Áp combo thử nghiệm
                candidate = dict(moves)
                for s, m in zip(involved, combo):
                    candidate[s.id] = m

                cand_nxt = self._next_positions(shippers, candidate)
                if self._has_conflict(shippers, cand_nxt):
                    continue

                score = self._move_score(involved, candidate, cand_nxt, orders, t)

                if score > best_score:
                    best_score  = score
                    best_combos = [combo]
                elif score == best_score:
                    best_combos.append(combo)

            if not best_combos:
                # Không tìm được tổ hợp hợp lệ → tất cả đứng yên
                for s in involved:
                    moves[s.id] = "S"
            else:
                chosen = self._rng.choice(best_combos)
                for s, m in zip(involved, chosen):
                    moves[s.id] = m

        return moves

    # ================================================================
    # State sync
    # ================================================================

    def _sync_sequences(self, obs: dict) -> None:
        """
        Đồng bộ self.sequences với trạng thái thực tế từ obs:
        - Xóa waypoint pickup cho đơn đã được nhặt (đang trong bag).
        - Xóa waypoints của đơn đã giao (không còn trong obs['orders']).
        - Bổ sung waypoint delivery cho đơn trong bag bị thiếu trong sequence.
        - Xóa dist_maps của đơn đã giao.
        """
        active = obs["orders"]   # chỉ chứa đơn chưa giao

        for s in obs["shippers"]:
            if s.id not in self.sequences:
                self.sequences[s.id] = []

            seq = self.sequences[s.id]

            # Xóa waypoints không còn hợp lệ
            seq = [
                (oid, is_p)
                for oid, is_p in seq
                if oid in active and not (is_p and oid in s.bag)
            ]

            # Bổ sung delivery waypoints cho đơn trong bag bị thiếu
            has_delivery = {oid for oid, is_p in seq if not is_p}
            for oid in s.bag:
                if oid in active and oid not in has_delivery:
                    seq.append((oid, False))

            self.sequences[s.id] = seq

        # Xóa dist_maps của đơn đã giao
        for oid in list(self.dist_maps):
            if oid not in active:
                del self.dist_maps[oid]

        # Xóa unassigned đã không còn tồn tại
        self.unassigned = {oid for oid in self.unassigned if oid in active}

    # ================================================================
    # Actions
    # ================================================================

    def _build_actions(
        self,
        shippers:    List[Shipper],
        final_moves: Dict[int, str],
        orders:      Dict[int, Order],
    ) -> Dict[int, Tuple[str, int]]:
        """
        Từ final_moves, xây dựng action (move, op) cho từng shipper.
        op = 1 nếu shipper vừa đến điểm pickup, op = 2 nếu đến điểm delivery.
        """
        actions: Dict[int, Tuple[str, int]] = {}

        for s in shippers:
            move    = final_moves.get(s.id, "S")
            new_pos = valid_next_pos(s.position, move, self.grid)
            seq     = self.sequences.get(s.id, [])
            op      = 0

            if seq:
                oid, is_pickup = seq[0]
                order = orders.get(oid)
                if order is not None:
                    goal = (order.sx, order.sy) if is_pickup else (order.ex, order.ey)
                    if new_pos == goal:
                        op = 1 if is_pickup else 2

            actions[s.id] = (move, op)

        return actions

    # ================================================================
    # Main decision (contract với run_visual.py)
    # ================================================================

    def _decide_actions(self, obs: dict) -> dict:
        """
        Nhận obs hiện tại, cập nhật state nội bộ và trả về actions
        cho tất cả shippers. Đây là contract chung với visualizer.
        """
        t        = obs["t"]
        orders   = obs["orders"]
        shippers = obs["shippers"]
        new_ids  = obs.get("new_order_ids", [])

        # 1. Đồng bộ sequences với trạng thái thực tế
        self._sync_sequences(obs)

        # 2. Build BFS maps cho đơn mới
        for oid in new_ids:
            if oid in orders and oid not in self.dist_maps:
                self._build_dist_maps(orders[oid])

        # 3. Phân công đơn mới cho shippers
        if new_ids:
            all_planned = {oid for seq in self.sequences.values() for oid, _ in seq}
            to_assign   = [oid for oid in new_ids
                           if oid in orders and oid not in all_planned]
            if to_assign:
                self._assign_new_orders(to_assign, shippers, orders, t)

        # 4. Thử lại phân công cho các đơn bị bỏ lỡ trước đó
        if self.unassigned:
            self._retry_unassigned(shippers, orders, t)

        # 5. Tính desired moves
        desired = {s.id: self._desired_move(s, orders) for s in shippers}

        # 6. Xử lý xung đột
        final_moves = self._resolve_conflicts(shippers, desired, orders, t)

        # 7. Build actions
        return self._build_actions(shippers, final_moves, orders)

    # ================================================================
    # Main loop
    # ================================================================

    def run(self) -> dict:
        start_time = time.time()
        obs        = self.env.reset()

        # Khởi tạo sequences và BFS maps cho đơn xuất hiện tại t=0
        for s in obs["shippers"]:
            self.sequences[s.id] = []
        for oid, order in obs["orders"].items():
            if oid not in self.dist_maps:
                self._build_dist_maps(order)

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
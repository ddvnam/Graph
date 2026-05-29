#region: Header
"""
greedy_bfs.py — Greedy BFS solver for Online MAPD.

Thuật toán (xem heuristic.txt để biết chi tiết):
  1. BFS maps   : Cache khoảng cách BFS từ mọi ô đến mọi điểm pickup/delivery của đơn hàng.
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
from collections import deque, Counter
from itertools import product
from typing import Dict, List, Set, Tuple
import heapq

from env import (
    DeliveryEnv, Order, Shipper,
    delivery_reward, move_cost, valid_next_pos
)
from solvers.solver import Solver

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Position = Tuple[int, int]
Waypoint = Tuple[int, bool] # (order_id, is_pickup)

INF       = 10 ** 9
MOVES_4 = ("U", "D", "L", "R")
MOVES_5 = ("S", "U", "D", "L", "R")
#endregion

class GreedyBFS(Solver):
    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.dist_maps: Dict[Position, List[List[int]]] = {} # dist_maps[goal_pos][i][j]
        self.sequences: Dict[int, List[Waypoint]] = {} # sequences[shipper_id]
        self.orders_map: Dict[Position, List[Tuple[int, int, int]]] = {}
        self.order_shipper: Dict[int, int] = {}
        self.unassigned: Set[int] = set()
        self.blacklist: Set[int] = set()
        self._rng = random.Random(42)

#region: BFS tính dist - XONG
    # ================================================================
    # BFS helpers
    # ================================================================

    def _bfs_from(self, source: Position) -> List[List[int]]:
        """
        BFS từ source trên grid, trả về ma trận 2D List[List[int]]
        với dist[x][y] = khoảng cách từ ô (x, y) đến source.
        """
        grid = self.grid
        N = self.env.N
        dist = [[INF] * N for _ in range(N)]
        dist[source[0]][source[1]] = 0
        q: deque[Position] = deque([source])
        while q:
            pos = q.popleft()
            d = dist[pos[0]][pos[1]]
            for m in MOVES_4:
                nxt = valid_next_pos(pos, m, grid)
                if nxt != pos and dist[nxt[0]][nxt[1]] == INF:
                    dist[nxt[0]][nxt[1]] = d + 1
                    q.append(nxt)
        return dist

    def _dist_to(self, current_pos: Position, goal_pos: Position) -> int:
        """
        Lấy khoảng cách BFS từ current_pos đến goal_pos. 
        Nếu goal_pos chưa được tính BFS, chạy và cache lại ngay.
        """
        if goal_pos not in self.dist_maps:
            self.dist_maps[goal_pos] = self._bfs_from(goal_pos)
        return self.dist_maps[goal_pos][current_pos[0]][current_pos[1]]
#endregion

#region: Tính hàm phần thưởng dự kiến - ĐÃ REVIEW
    # ================================================================
    # Reward estimation
    # ================================================================

    def _estimate_reward(
        self,
        start_pos:  Position,
        start_bag:  List[int],
        sequence:   List[Waypoint],
        orders:     Dict[int, Order],
        shipper:    Shipper,
        t_now:      int,
    ) -> Tuple[float, Set[int]]:
        """
        Ước lượng tổng phần thưởng tương lai của shipper khi theo sequence.
        Coi như chỉ mình shipper đó trên map (không tính xung đột).
        """
        pos = start_pos
        t = t_now
        T = self.env.T
        W = shipper.W_max
        w = sum(orders[bid].w for bid in start_bag)
        reward = 0.0
        released_orders: Set[int] = set()
        sequence_iter = iter(sequence)
        for oid, is_pickup in sequence_iter:
            order = orders[oid]
            goal = (order.sx, order.sy) if is_pickup else (order.ex, order.ey)
            d = self._dist_to(pos, goal)
            nxt_t = t + (1 if d == 0 and is_pickup else d)
            if (nxt_t > T):
                reward += (T - t) * move_cost(w, W)
                for r_oid, r_is_pickup in sequence_iter:
                    if not r_is_pickup: released_orders.add(r_oid)
                break
            t = nxt_t
            reward += d * move_cost(w, W) # move_cost trả về giá trị âm
            pos = goal
            if is_pickup:
                w += order.w
            else:
                w -= order.w
                reward += delivery_reward(order, t, T)
        return reward, released_orders
#endregion

    def _is_feasible(
        self,
        start_bag: List[int],
        sequence:  List[Waypoint],
        orders:    Dict[int, Order],
        shipper:   Shipper,
    ) -> bool:
        """
        True nếu sequence không vi phạm K_max / W_max tại bất kỳ bước pickup nào.
        Không quan tâm đến thời gian.
        """
        W = shipper.W_max
        K = shipper.K_max
        w = sum(orders[bid].w for bid in start_bag)
        k = len(start_bag)
        for oid, is_pickup in sequence:
            order = orders[oid]
            if not is_pickup:
                w -= order.w
                k -= 1
            else:
                w += order.w
                k += 1
                if k > K or w > W: return False
        return True

#region: Phân công đơn hàng - CHƯA REVIEW
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
    ) -> Tuple[float, List[Waypoint], Set[int]]:
        """
        Thử tất cả vị trí chèn hợp lệ (p_idx, d_idx) cho
        (pickup_waypoint, delivery_waypoint) của order vào sequence.

        Với sequence có k waypoint:
          - pickup  tại p_idx ∈ [0, k]
          - delivery tại d_idx ∈ [p_idx+1, k+1]  (sau khi đã chèn pickup)
        Tổng: (k+1)(k+2)/2 tổ hợp.

        Trả (best_gain, best_sequence, released_orders) hoặc None nếu không có insertion hợp lệ.
        """
        k = len(sequence)
        baseline_reward, _ = self._estimate_reward(pos, start_bag, sequence, orders, shipper, t_now)
        best_reward = baseline_reward
        # if (k >= 20): return 0, sequence, [order.id]
        best_sequences: List[Tuple[List[Waypoint], List[Order]]] = []
        for p in range(k + 1):
            temp = sequence[:p] + [(order.id, True)] + sequence[p:]
            for d in range(p + 1, k + 2):
                new_sequence = temp[:d] + [(order.id, False)] + temp[d:]
                reward, released_orders = self._estimate_reward(pos, start_bag, new_sequence, orders, shipper, t_now)
                if released_orders:
                    new_sequence = [(oid, is_pickup) for oid, is_pickup in new_sequence if oid not in released_orders]
                    reward, _ = self._estimate_reward(pos, start_bag, new_sequence, orders, shipper, t_now)
                if not self._is_feasible(start_bag, new_sequence, orders, shipper): continue
                if reward > best_reward:
                    best_reward  = reward
                    best_sequences = [(new_sequence, released_orders)]
                elif reward == best_reward:
                    best_sequences.append((new_sequence, released_orders))
        if not best_sequences: return 0, sequence, [order.id]
        chosen = self._rng.choice(best_sequences)
        return best_reward - baseline_reward, chosen[0], chosen[1]

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
        Phân công tuần tự (Greedy Sequential) cho toàn bộ các đơn đang mở (visible unassigned).
        Bao gồm cả đơn mới xuất hiện hợp lệ và đơn bị các shipper nhả ra.
        """
        for oid in new_ids:
            order = orders[oid]
            pickup_pos = (order.sx, order.sy)
            
            if pickup_pos not in self.orders_map:
                self.orders_map[pickup_pos] = []
            queue = self.orders_map[pickup_pos]
            
            if queue:
                _, _, old_top_oid = queue[0]
                new_tuple = (-order.p, order.et, order.id)
                if new_tuple < queue[0]:
                    if old_top_oid in self.order_shipper:
                        sid = self.order_shipper[old_top_oid]
                        self.sequences[sid] = [(x, is_p) for x, is_p in self.sequences[sid] if x != old_top_oid]
                        del self.order_shipper[old_top_oid]
                    else:
                        self.unassigned.discard(old_top_oid)
                    
            heapq.heappush(queue, (-order.p, order.et, order.id))

        for _, queue in self.orders_map.items():
            if queue: self.unassigned.add(queue[0][2])
        self.unassigned = {oid for oid in self.unassigned if oid not in self.blacklist and oid not in self.order_shipper}

        working_pool = sorted(self.unassigned, key=lambda oid: (-orders[oid].p, orders[oid].et, oid))
        # print("un", len(working_pool))

        for oid in working_pool:   
            order = orders[oid]
            best_gain = 0
            best_shippers = []
            for s in shippers:
                gain, seq, released = self._best_insert( self.sequences[s.id], order, s.bag, orders, s, s.position, t)
                if gain > best_gain:
                    best_gain = gain
                    best_shippers = [(s, seq, released)]
                elif gain == best_gain:
                    best_shippers.append((s, seq, released))
            if best_gain > 0 and best_shippers:
                s, seq, released = self._rng.choice(best_shippers)
                self.sequences[s.id] = seq
                self.order_shipper[oid] = s.id
                self.unassigned.discard(oid)
                for r_oid in released:
                    self.unassigned.add(r_oid)
                    del self.order_shipper[r_oid]
            else:
                self.blacklist.add(oid)
        
        # print("max seq", max(len(seq) for seq in self.sequences.values()))

        # for oid in orders:
        #     if oid not in self.order_shipper: print(orders[oid].w, '(', orders[oid].p, ')', orders[oid].sx, orders[oid].sy)
        # print("-----")
#endregion

#region: Xử lý xung đột di chuyển - ĐÃ REVIEW
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
        seq = self.sequences[shipper.id]
        if not seq: return "S"
        oid, is_pickup = seq[0]
        order = orders[oid]
        goal = (order.sx, order.sy) if is_pickup else (order.ex, order.ey)
        if shipper.position == goal: return "S"
        pos = shipper.position
        d = self._dist_to(pos, goal)
        best_moves: List[str] = []
        for m in ("U", "D", "L", "R"):
            nxt = valid_next_pos(pos, m, self.grid)
            if nxt != pos and self._dist_to(nxt, goal) < d: best_moves.append(m)
        return self._rng.choice(best_moves)

    # ================================================================
    # Conflict detection & resolution
    # ================================================================

    def _next_positions(
        self,
        shippers: List[Shipper],
        moves:    Dict[int, str],
    ) -> Dict[int, Position]:
        return {
            s.id: valid_next_pos(s.position, moves[s.id], self.grid)
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

    def _conflicted_shippers(
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
            total  += self._estimate_reward(new_pos, s.bag, seq, orders, s, t + 1)[0]
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
        """
        moves = dict(desired)
        nxt_map = self._next_positions(shippers, moves)
        if not self._has_conflict(shippers, nxt_map): return moves
        involved = self._conflicted_shippers(shippers, nxt_map)
        if not involved: return moves
        if len(involved) > 5:
            for s in involved: moves[s.id] = self._rng.choice(MOVES_5)
            return moves
        best_score = -float("inf")
        best_combos: List[Tuple] = []
        for combo in product(MOVES_5, repeat=len(involved)):
            candidate = dict(moves)
            is_valid = True
            for s, m in zip(involved, combo):
                if m != "S" and valid_next_pos(s.position, m, self.grid) == s.position:
                    is_valid = False
                    break
                candidate[s.id] = m
            if not is_valid: continue
            cand_nxt = self._next_positions(shippers, candidate)
            if combo.count("S") == len(involved): continue
            if self._has_conflict(shippers, cand_nxt): continue
            score = self._move_score(involved, candidate, cand_nxt, orders, t)
            if score > best_score:
                best_score  = score
                best_combos = [combo]
            elif score == best_score:
                best_combos.append(combo)
        chosen = self._rng.choice(best_combos)
        for s, m in zip(involved, chosen): moves[s.id] = m
        return moves
#endregion

#region: Code tương tác với môi trường
    # ================================================================
    # State sync
    # ================================================================

    def _sync_sequences(self, obs: dict) -> None:
        """
        Đồng bộ toàn bộ các attribute với trạng thái thực tế từ obs sau các action ở tick trước.
        """
        active = obs["orders"]   # Chỉ chứa đơn chưa giao
        all_bags = set()         # Tập hợp tất cả đơn đã nằm trên xe của các shipper
        
        # --- BƯỚC 1: Đồng bộ sequences và thu thập đơn trong bag ---
        for s in obs["shippers"]:
            all_bags.update(s.bag)
            if s.id not in self.sequences:
                self.sequences[s.id] = []

            seq = self.sequences[s.id]

            # Xóa waypoints không còn hợp lệ (đơn đã giao hoặc điểm pickup của đơn đã được nhặt)
            self.sequences[s.id] = [
                (oid, is_p)
                for oid, is_p in seq
                if oid in active and not (is_p and oid in s.bag)
            ]

        # --- BƯỚC 2: Cập nhật order_shipper ---
        # Xóa các đơn đã hoàn thành giao hàng (không còn trong active)
        for oid in list(self.order_shipper.keys()):
            if oid not in active:
                del self.order_shipper[oid]

        # --- BƯỚC 3: Cập nhật orders_map ---
        # Lọc bỏ các đơn đã được nhặt vào bag hoặc đơn đã giao xong khỏi hàng đợi các ô
        for pos, queue in list(self.orders_map.items()):
            # Chỉ giữ lại các đơn chưa giao VÀ chưa bị shipper nào nhặt lên xe
            updated_queue = [item for item in queue if item[2] in active and item[2] not in all_bags]
            
            if updated_queue:
                heapq.heapify(updated_queue) # Re-heapify lại danh sách sau khi lọc
                self.orders_map[pos] = updated_queue
            else:
                del self.orders_map[pos] # Ô không còn đơn nào thì xóa hẳn key khỏi map

        # --- BƯỚC 4: Dọn dẹp unassigned (BẮT BUỘC GIỮ) ---
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

        # 2. Phân công đơn mới cho shippers
        self._assign_new_orders(new_ids, shippers, orders, t)

        # 3. Tính desired moves
        desired = {s.id: self._desired_move(s, orders) for s in shippers}

        # 4. Xử lý xung đột
        final_moves = self._resolve_conflicts(shippers, desired, orders, t)

        # print("-----")
        # print("t", obs["t"])

        # 5. Build actions
        return self._build_actions(shippers, final_moves, orders)

    # ================================================================
    # Main loop
    # ================================================================

    def run(self) -> dict:
        start_time = time.time()
        obs        = self.env.reset()

        # Khởi tạo sequences cho đơn xuất hiện tại t=0
        for s in obs["shippers"]:
            self.sequences[s.id] = []

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
#endregion
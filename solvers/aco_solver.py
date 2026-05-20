from __future__ import annotations

import time
import random
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple, Any

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")

class ACOSolver(Solver):
    """
    ACO Solver kết hợp Cơ chế Chống Deadlock Chủ động.
    """

    method_name = "ACO_Solver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        
        # Hyperparameters cho ACO
        self.num_ants = 5
        self.aco_iterations = 5
        self.alpha = 1.0          
        self.beta = 2.0           
        self.evaporation_rate = 0.3
        self.q = 10.0             
        
        self.pheromones: Dict[int, Dict[str, float]] = {}
        self.stuck_counters: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Thuật toán Pathfinding & Heuristics
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(self, start: Position, goal: Position) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None
        queue: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}
        while queue:
            current = queue.popleft()
            if current == goal: return parent
            for move, nxt in self._neighbors(current):
                if nxt not in parent:
                    parent[nxt] = (current, move)
                    queue.append(nxt)
        return None

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal: return 0
        key = (start, goal)
        if key in self._distance_cache: return self._distance_cache[key]
        parent = self._bfs_parents(start, goal)
        dist = INF
        if parent and goal in parent:
            dist = 0
            curr = goal
            while curr != start:
                prev, _ = parent[curr]
                if prev is None: break
                curr = prev
                dist += 1
        self._distance_cache[key] = dist
        return dist

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal: return "S"
        key = (start, goal)
        if key in self._next_move_cache: return self._next_move_cache[key]
        parent = self._bfs_parents(start, goal)
        move = "S"
        if parent and goal in parent:
            curr = goal
            while True:
                prev, m = parent[curr]
                if prev is None: break
                if prev == start:
                    move = m
                    break
                curr = prev
        self._next_move_cache[key] = move
        return move

    def _get_valid_tasks(self, shipper: Shipper, orders: Dict[int, Order], claimed_pickups: set[int]) -> List[str]:
        valid = []
        for oid in shipper.bag:
            if oid in orders and not orders[oid].delivered:
                valid.append(f"D_{oid}")
        current_w = sum(orders[i].w for i in shipper.bag if i in orders)
        for oid, order in orders.items():
            if order.picked or order.delivered or (oid in claimed_pickups): continue
            if len(shipper.bag) < shipper.K_max and current_w + order.w <= shipper.W_max:
                valid.append(f"P_{oid}")
        return valid

    def _heuristic(self, shipper: Shipper, task_id: str, orders: Dict[int, Order], current_t: int) -> float:
        task_type, oid_str = task_id.split("_")
        order = orders[int(oid_str)]
        target_pos = (order.sx, order.sy) if task_type == "P" else (order.ex, order.ey)
        dist = self._distance(shipper.position, target_pos)
        if dist >= INF: return 0.0001
        
        time_left = max(1, order.et - current_t - dist)
        urgency = 100.0 / time_left if time_left > 0 else 0.1
        
        base_reward = 4.0
        if order.w > 30: base_reward = 30.0
        elif order.w > 10: base_reward = 20.0
        elif order.w > 3: base_reward = 15.0
        elif order.w > 0.2: base_reward = 10.0
        
        reward = base_reward * {1: 1.0, 2: 2.0, 3: 3.0}.get(order.p, 1.0)
        return (reward / (dist + 1.0)) + urgency

    # ------------------------------------------------------------------
    # Thuật toán ACO
    # ------------------------------------------------------------------
    def _run_aco(self, obs: dict) -> Dict[int, str]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        current_t = obs["t"]

        all_tasks = [f"D_{o.id}" for o in orders.values()] + [f"P_{o.id}" for o in orders.values()]
        for s in shippers:
            if s.id not in self.pheromones: self.pheromones[s.id] = {}
            for t in all_tasks:
                if t not in self.pheromones[s.id]: self.pheromones[s.id][t] = 1.0

        best_assignment = {}
        best_score = -1.0

        for _ in range(self.aco_iterations):
            ant_assignments = []
            ant_scores = []
            for _ in range(self.num_ants):
                assignment = {}
                claimed_pickups = set()
                score = 0.0

                for s in shippers:
                    valid_tasks = self._get_valid_tasks(s, orders, claimed_pickups)
                    if not valid_tasks: continue

                    probs = []
                    for t in valid_tasks:
                        tau = self.pheromones[s.id].get(t, 1.0)
                        eta = self._heuristic(s, t, orders, current_t)
                        weight = (tau ** self.alpha) * (eta ** self.beta)
                        probs.append((t, weight, eta))

                    total_weight = sum(p[1] for p in probs)
                    if total_weight == 0:
                        chosen_task, _, chosen_eta = random.choice(probs)
                    else:
                        r = random.uniform(0, total_weight)
                        cumulative = 0.0
                        for t, w, eta in probs:
                            cumulative += w
                            if r <= cumulative:
                                chosen_task, chosen_eta = t, eta
                                break
                        else:
                            chosen_task, _, chosen_eta = probs[-1]

                    assignment[s.id] = chosen_task
                    score += chosen_eta
                    if chosen_task.startswith("P_"):
                        claimed_pickups.add(int(chosen_task.split("_")[1]))

                ant_assignments.append(assignment)
                ant_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_assignment = assignment

            for sid in self.pheromones:
                for t in self.pheromones[sid]:
                    self.pheromones[sid][t] *= (1.0 - self.evaporation_rate)

            if ant_scores:
                best_iter_idx = ant_scores.index(max(ant_scores))
                for sid, t in ant_assignments[best_iter_idx].items():
                    self.pheromones[sid][t] += self.q * ant_scores[best_iter_idx]

        return best_assignment

    # ------------------------------------------------------------------
    # Cơ chế lách đường (Evasion) chủ động
    # ------------------------------------------------------------------
    def _resolve_conflicts(self, shippers: List[Shipper], desired_moves: Dict[int, Action]) -> Dict[int, Action]:
        moves = {s.id: desired_moves.get(s.id, ("S", 0))[0] for s in shippers}
        ops = {s.id: desired_moves.get(s.id, ("S", 0))[1] for s in shippers}

        final_moves = {}
        claimed_pos = set()
        
        # Ưu tiên tính đường cho những shipper bị kẹt lâu nhất
        order = sorted(shippers, key=lambda s: (-self.stuck_counters.get(s.id, 0), s.id))
        
        for s in order:
            sid = s.id
            intended_move = moves[sid]
            nxt = valid_next_pos(s.position, intended_move, self.grid)
            
            conflict = False
            # 1. Vertex Collision: Ô đích đã bị shipper khác ưu tiên chiếm trước
            if nxt in claimed_pos and nxt != s.position:
                conflict = True
                
            # 2. Block Collision: Ô đích đang có shipper khác đứng (mà người đó không di chuyển đi)
            if not conflict:
                for other in shippers:
                    if other.id != sid and other.position == nxt:
                        if other.id not in final_moves or final_moves[other.id] == "S":
                            conflict = True
                            break
                            
            # 3. Swap Collision: Hai người đang định đổi chỗ trực tiếp cho nhau
            if not conflict:
                for other in shippers:
                    if other.id != sid and other.id in final_moves:
                        other_nxt = valid_next_pos(other.position, final_moves[other.id], self.grid)
                        if other_nxt == s.position and nxt == other.position:
                            conflict = True
                            break

            if conflict:
                # Evasion: Chủ động quét các ô lân cận để lách
                possible_moves = []
                for m in MOVES:
                    alt_nxt = valid_next_pos(s.position, m, self.grid)
                    if alt_nxt != s.position and alt_nxt not in claimed_pos:
                        occ_safe = True
                        for other in shippers:
                            if other.id != sid and other.position == alt_nxt:
                                if other.id not in final_moves or final_moves[other.id] == "S":
                                    occ_safe = False
                                    break
                        if occ_safe:
                            possible_moves.append(m)
                
                # Nếu có hướng lách an toàn, tự động chuyển hướng. Nếu kẹt cứng, đành đứng im chờ.
                if possible_moves:
                    final_moves[sid] = random.choice(possible_moves)
                else:
                    final_moves[sid] = "S"
            else:
                final_moves[sid] = intended_move
                
            claimed_pos.add(valid_next_pos(s.position, final_moves[sid], self.grid))

        actions = {}
        for s in shippers:
            sid = s.id
            
            # Nếu buộc phải đổi hướng lách đi chỗ khác, hủy thao tác (op=0) để tránh giao/nhận nhầm tọa độ
            if final_moves[sid] != moves[sid] and final_moves[sid] != "S":
                actions[sid] = (final_moves[sid], 0)
            else:
                actions[sid] = (final_moves[sid], ops[sid])
            
            # Cập nhật số turn bị kẹt
            if final_moves[sid] == "S" and desired_moves.get(sid, ("S", 0))[0] != "S":
                self.stuck_counters[sid] = self.stuck_counters.get(sid, 0) + 1
            else:
                self.stuck_counters[sid] = 0
                
        return actions

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------
    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        assignments = self._run_aco(obs)
        desired_moves: Dict[int, Action] = {}
        
        for s in shippers:
            task_id = assignments.get(s.id)
            if not task_id:
                desired_moves[s.id] = ("S", 0)
                continue

            task_type, oid_str = task_id.split("_")
            order = orders[int(oid_str)]
            target_pos = (order.sx, order.sy) if task_type == "P" else (order.ex, order.ey)

            move = self._next_move(s.position, target_pos)
            nxt_pos = valid_next_pos(s.position, move, self.grid)

            op = 0
            if nxt_pos == target_pos:
                op = 1 if task_type == "P" else 2

            desired_moves[s.id] = (move, op)

        return self._resolve_conflicts(shippers, desired_moves)

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()
        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done: break
        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
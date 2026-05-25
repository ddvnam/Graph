"""
run_visual.py — Visualize DeliveryEnv với solver tùy chọn

Chạy:
    python run_visual.py --config test_config.txt --method GreedyBFS [--seed 42] [--speed 5] [--config-index 0]

Điều khiển trong cửa sổ:
    SPACE       — tạm dừng / tiếp tục
    → / ←       — tăng / giảm tốc độ
    R           — reset lại từ đầu
    Q / Esc     — thoát
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import os
import sys
import time
import math
from typing import Any, Dict, List, Optional, Tuple

import pygame

# ---------------------------------------------------------------------------
# Paths — thêm thư mục chứa env.py và solvers/ vào sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SOLVER_DIR = os.path.join(SCRIPT_DIR, "solvers")
for p in [SCRIPT_DIR, BASE_SOLVER_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from env import DeliveryEnv, Order, Shipper, SEED, load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG_COLOR          = (30,  30,  40)
GRID_COLOR        = (55,  55,  70)
CELL_FREE         = (220, 220, 230)
CELL_OBSTACLE     = (60,  60,  70)
PANEL_BG          = (20,  20,  30)
TEXT_COLOR        = (230, 230, 230)
TEXT_DIM          = (140, 140, 160)

SHIPPER_PALETTE = [
    (52, 152, 219),   # xanh dương
    (231, 76,  60),   # đỏ
    (46,  204, 113),  # xanh lá
    (155,  89, 182),  # tím
    (243, 156,  18),  # cam
    (26,  188, 156),  # ngọc
    (236,  72, 153),  # hồng
    (99,  110, 114),  # xám
]

PICKUP_COLOR   = (39,  174,  96)   # xanh lá — ô lấy hàng
DELIVERY_COLOR = (231,  76,  60)   # đỏ      — ô giao hàng
CARRIED_COLOR  = (243, 156,  18)   # cam      — đơn đang mang

PRIORITY_COLOR = {1: (189, 195, 199), 2: (52, 152, 219), 3: (231, 76, 60)}

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
PANEL_WIDTH   = 320
MIN_CELL      = 16
MAX_CELL      = 56
PADDING       = 12

# ---------------------------------------------------------------------------
# Solver loader (giống run_test.py)
# ---------------------------------------------------------------------------
SOLVER_SOURCES = [
    ("GreedyBFS",        "greedy_bfs.py"),
    ("VRPOrToolsSolver", "vrp_ortools.py"),
    ("ACOSolver",        "aco_solver.py"),
    ("MAPDCBSSolver",    "mapd_cbs_solver.py"),
    ("CustomSolver",     "custom_solver.py"),
    ("Baseline",         "baseline.py"),
]


def load_solver_class(class_name: str, file_name: str):
    path = os.path.join(BASE_SOLVER_DIR, file_name)
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Không tìm thấy {file_name} trong thư mục solvers/")
    spec = importlib.util.spec_from_file_location(class_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name, None)
    if cls is None:
        sys.exit(f"[ERROR] Không tìm thấy lớp {class_name} trong {file_name}")
    return cls


def get_solver_class(method: str):
    for name, fname in SOLVER_SOURCES:
        if name == method:
            return load_solver_class(name, fname)
    available = [n for n, _ in SOLVER_SOURCES]
    sys.exit(f"[ERROR] Phương pháp '{method}' không tồn tại. Các phương pháp: {', '.join(available)}")


def stable_config_seed(config_name: str, base_seed: int) -> int:
    digest = hashlib.md5(f"{base_seed}:{config_name}".encode()).hexdigest()
    return int(digest[:8], 16)


# ---------------------------------------------------------------------------
# Stepping solver — wrapper chạy từng bước một thay vì chạy toàn bộ
# ---------------------------------------------------------------------------
class SteppingSolver:
    """Bọc solver gốc: thay vì gọi solver.run(), ta gọi decide() từng bước."""

    def __init__(self, solver_cls, cfg: dict, seed: int):
        self.cfg        = copy.deepcopy(cfg)
        self.seed       = seed
        self.solver_cls = solver_cls
        self.next_actions = {}
        self._build()

    def _build(self):
        self.env    = DeliveryEnv(copy.deepcopy(self.cfg), seed=self.seed)
        self.solver = self.solver_cls(self.env)
        self.obs    = self.env.reset()
        self.done   = self.obs.get("done", False)
        self._peek_actions()

    def _peek_actions(self):
        """Tính toán trước hành động cho bước tiếp theo để phục vụ hiển thị."""
        if not self.done:
            self.next_actions = self.solver._decide_actions(self.obs)
        else:
            self.next_actions = {}

    def reset(self):
        self._build()

    def step(self) -> dict:
        """Chạy một bước; trả obs mới."""
        if self.done:
            return self.obs
        actions  = self.next_actions
        self.obs, _, done, _ = self.env.step(actions)
        self.done = done
        self._peek_actions()
        return self.obs

    def result(self) -> dict:
        return self.env.result(self.solver.method_name)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class Renderer:
    def __init__(
        self,
        configs: List[dict],
        solver_cls,
        base_seed: int,
        start_index: int = 0,
        speed: int = 5,
    ):
        self.configs    = configs
        self.solver_cls = solver_cls
        self.base_seed  = base_seed
        self.cfg_index  = start_index
        self.speed      = max(1, speed)
        self.paused     = False

        self.stepping = self._make_stepping(self.cfg_index)

        obs  = self.stepping.obs
        grid = obs["grid"]
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 1

        pygame.init()
        pygame.display.set_caption("Delivery Env Visualizer")

        # Tính kích thước cell sao cho vừa màn hình
        disp     = pygame.display.Info()
        max_w    = disp.current_w  - PANEL_WIDTH - PADDING * 3
        max_h    = disp.current_h  - PADDING * 2 - 60
        cell     = min(MAX_CELL, max_w // self.cols, max_h // self.rows)
        cell     = max(MIN_CELL, cell)
        self.cell = cell

        grid_w = cell * self.cols
        grid_h = cell * self.rows
        win_w  = grid_w + PANEL_WIDTH + PADDING * 3
        win_h  = max(grid_h + PADDING * 2, 500) + 60   # +60 cho thanh status dưới

        self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        self.font_s = pygame.font.SysFont("monospace", max(9,  cell // 3))
        self.font_m = pygame.font.SysFont("monospace", max(11, cell // 2))
        self.font_l = pygame.font.SysFont("monospace", 16)
        self.font_h = pygame.font.SysFont("monospace", 18, bold=True)

        self.grid_offset_x = PADDING
        self.grid_offset_y = PADDING

        self.clock       = pygame.time.Clock()
        self._last_step  = time.time()
        self.events_log: List[str] = []   # log sự kiện gần nhất

    # ------------------------------------------------------------------
    # Config navigation helpers
    # ------------------------------------------------------------------
    def _make_stepping(self, idx: int) -> SteppingSolver:
        cfg  = self.configs[idx]
        seed = stable_config_seed(str(cfg.get("name", "unknown")), self.base_seed)
        return SteppingSolver(self.solver_cls, cfg, seed)

    def _switch_config(self, new_idx: int):
        """Chuyển sang config new_idx, rebuild stepping và cập nhật tiêu đề."""
        self.cfg_index = new_idx % len(self.configs)
        self.stepping  = self._make_stepping(self.cfg_index)
        cfg = self.configs[self.cfg_index]
        pygame.display.set_caption(
            f"Delivery Visualizer — {cfg.get('name','?')} "
            f"({self.cfg_index + 1}/{len(self.configs)})"
        )
        self._last_step = time.time()
        print(f"[Config {self.cfg_index}] {cfg.get('name','?')}  N={cfg['N']} C={cfg['C']} G={cfg['G']} T={cfg['T']}")

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def cell_rect(self, r: int, c: int) -> pygame.Rect:
        x = self.grid_offset_x + c * self.cell
        y = self.grid_offset_y + r * self.cell
        return pygame.Rect(x, y, self.cell, self.cell)

    def cell_center(self, r: int, c: int) -> Tuple[int, int]:
        rect = self.cell_rect(r, c)
        return rect.centerx, rect.centery

    # ------------------------------------------------------------------
    # Draw grid
    # ------------------------------------------------------------------
    def draw_grid(self, grid: List[List[int]]):
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                rect  = self.cell_rect(r, c)
                color = CELL_OBSTACLE if val else CELL_FREE
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

    # ------------------------------------------------------------------
    # Draw orders
    # ------------------------------------------------------------------
    def draw_orders(self, orders: Dict[int, Order], shippers: List[Shipper]):
        carried_ids = {oid for s in shippers for oid in s.bag}
        cell = self.cell
        pad  = max(2, cell // 8)

        for order in orders.values():
            if order.delivered:
                continue

            p_color = PRIORITY_COLOR.get(order.p, TEXT_COLOR)

            if not order.picked:
                # Ô lấy hàng — hình vuông viền xanh lá
                r, c = order.sx, order.sy
                rect = self.cell_rect(r, c).inflate(-pad * 2, -pad * 2)
                pygame.draw.rect(self.screen, PICKUP_COLOR, rect, max(2, cell // 10))
                # Chấm ưu tiên
                cx, cy = self.cell_center(r, c)
                pygame.draw.circle(self.screen, p_color, (cx, cy), max(3, cell // 7))

                # Ô giao hàng — mũi tên / hình tròn viền đỏ (mờ)
                er, ec = order.ex, order.ey
                dr = self.cell_rect(er, ec).inflate(-pad * 2, -pad * 2)
                surf = pygame.Surface((dr.width, dr.height), pygame.SRCALPHA)
                surf.fill((*DELIVERY_COLOR, 50))
                self.screen.blit(surf, dr.topleft)
                pygame.draw.rect(self.screen, DELIVERY_COLOR, dr, max(1, cell // 14))

            elif order.id in carried_ids:
                # Đơn đang được mang — tô nhẹ ô đích
                er, ec = order.ex, order.ey
                dr = self.cell_rect(er, ec).inflate(-pad * 2, -pad * 2)
                surf = pygame.Surface((dr.width, dr.height), pygame.SRCALPHA)
                surf.fill((*CARRIED_COLOR, 80))
                self.screen.blit(surf, dr.topleft)
                pygame.draw.rect(self.screen, CARRIED_COLOR, dr, max(2, cell // 10))

    # ------------------------------------------------------------------
    # Draw arrows for shippers direction
    # ------------------------------------------------------------------
    def _draw_arrow(self, cx: float, cy: float, radius: float, move: str, color: tuple):
        vec_map = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
        if move not in vec_map:
            return
            
        dx, dy = vec_map[move]
        
        # Điểm bắt đầu từ viền ngoài vòng tròn shipper
        start_x = cx + dx * radius
        start_y = cy + dy * radius
        
        # Chiều dài mũi tên bằng một nửa kích thước cell
        arrow_len = self.cell / 2.0
        end_x = start_x + dx * arrow_len
        end_y = start_y + dy * arrow_len
        
        # Vẽ thân mũi tên
        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), max(2, self.cell // 10))
        
        # Vẽ tam giác làm đầu mũi tên
        head_size = max(3, self.cell // 8)
        nx, ny = -dy, dx  # Vector pháp tuyến vuông góc với hướng đi
        
        p1 = (end_x, end_y)
        p2 = (end_x - dx * head_size + nx * head_size, end_y - dy * head_size + ny * head_size)
        p3 = (end_x - dx * head_size - nx * head_size, end_y - dy * head_size - ny * head_size)
        pygame.draw.polygon(self.screen, color, [p1, p2, p3])

    # ------------------------------------------------------------------
    # Draw shippers
    # ------------------------------------------------------------------
    def draw_shippers(self, shippers: List[Shipper], orders: Dict[int, Order], next_actions: dict):
        cell   = self.cell
        radius = max(5, cell // 3)

        for shipper in shippers:
            color = SHIPPER_PALETTE[shipper.id % len(SHIPPER_PALETTE)]
            cx, cy = self.cell_center(shipper.r, shipper.c)

            # Vòng ngoài (tải trọng)
            w_max     = shipper.W_max
            w_carried = sum(orders[oid].w for oid in shipper.bag if oid in orders)
            load_frac = min(1.0, w_carried / max(w_max, 1e-9))
            outer_r   = radius + 4
            # Vẽ cung tải trọng
            if load_frac > 0:
                arc_surf = pygame.Surface((outer_r * 2 + 2, outer_r * 2 + 2), pygame.SRCALPHA)
                arc_rect = pygame.Rect(0, 0, outer_r * 2, outer_r * 2)
                end_angle = -math.pi / 2 + load_frac * 2 * math.pi
                pygame.draw.arc(
                    arc_surf, (*CARRIED_COLOR, 200), arc_rect,
                    -math.pi / 2, end_angle, max(3, cell // 8)
                )
                self.screen.blit(arc_surf, (cx - outer_r, cy - outer_r))

            # Thân shipper
            pygame.draw.circle(self.screen, color, (cx, cy), radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), radius, 2)

            # ID
            label = self.font_s.render(str(shipper.id), True, (255, 255, 255))
            self.screen.blit(label, label.get_rect(center=(cx, cy)))

            # Số đơn trong bag
            if shipper.bag:
                badge_r = max(5, cell // 5)
                bx, by  = cx + radius - 1, cy - radius + 1
                pygame.draw.circle(self.screen, (255, 60, 60), (bx, by), badge_r)
                cnt = self.font_s.render(str(len(shipper.bag)), True, (255, 255, 255))
                self.screen.blit(cnt, cnt.get_rect(center=(bx, by)))

            # Vẽ mũi tên biểu diễn hướng di chuyển tiếp theo
            move = next_actions.get(shipper.id, ("S", 0))[0]
            if move != "S":
                self._draw_arrow(cx, cy, radius, move, color)

    # ------------------------------------------------------------------
    # Draw side panel
    # ------------------------------------------------------------------
    def draw_panel(self, obs: dict, result_dict: Optional[dict] = None):
        w, h = self.screen.get_size()
        px     = self.grid_offset_x + self.cols * self.cell + PADDING
        panel  = pygame.Rect(px, 0, PANEL_WIDTH, h)
        pygame.draw.rect(self.screen, PANEL_BG, panel)

        t       = obs["t"]
        T       = obs["T"]
        G       = obs["G"]
        orders  = obs["orders"]
        shippers = obs["shippers"]

        picked_count    = sum(1 for o in orders.values() if o.picked and not o.delivered)
        waiting_count   = sum(1 for o in orders.values() if not o.picked and not o.delivered)
        info = self.stepping.env.info()

        lines: List[Tuple[str, Any, Any]] = [
            ("═" * 28,       None,           TEXT_DIM),
            ("TRẠNG THÁI",   None,           TEXT_COLOR),
            (f"Config: {self.configs[self.cfg_index].get('name','?')} ({self.cfg_index+1}/{len(self.configs)})", None, (243, 156, 18)),
            ("═" * 28,       None,           TEXT_DIM),
            (f"Thời gian",   f"{t} / {T}",   TEXT_COLOR),
            (f"Đơn tổng (G)",f"{G}",         TEXT_COLOR),
            (f"Đã sinh",     f"{info['generated']}", TEXT_COLOR),
            (f"Chờ nhặt",    f"{waiting_count}",     PICKUP_COLOR),
            (f"Đang mang",   f"{picked_count}",       CARRIED_COLOR),
            (f"Đã giao",     f"{info['delivered']}",  (46, 204, 113)),
            (f"  Đúng hạn",  f"{info['on_time']}",    (46, 204, 113)),
            (f"  Trễ hạn",   f"{info['late']}",       (231, 76, 60)),
            (f"Bỏ lỡ",       f"{info['missed']}",     (231, 76, 60)),
            ("─" * 28,       None,           TEXT_DIM),
            (f"Net reward",  f"{info['net_reward']:.2f}",     (243, 156, 18)),
            (f"Move cost",   f"{info['total_movecost']:.2f}", TEXT_DIM),
            ("─" * 28,       None,           TEXT_DIM),
        ]

        # Shipper info
        lines.append(("SHIPPERS",  None, TEXT_COLOR))
        for s in shippers:
            w_carried = sum(orders[oid].w for oid in s.bag if oid in orders)
            color     = SHIPPER_PALETTE[s.id % len(SHIPPER_PALETTE)]
            lines.append((
                f"  #{s.id} ({s.r},{s.c}) bag={len(s.bag)}/{s.K_max}",
                f"w={w_carried:.1f}/{s.W_max:.0f}",
                color,
            ))

        lines.append(("─" * 28, None, TEXT_DIM))

        # Controls
        speed_label = f"Tốc độ: {self.speed} bước/s"
        pause_label = "[SPACE] Tạm dừng" if not self.paused else "[SPACE] Tiếp tục"
        lines += [
            ("ĐIỀU KHIỂN",  None,          TEXT_DIM),
            (pause_label,   None,          TEXT_DIM),
            (speed_label,   "← →",        TEXT_DIM),
            ("[R] Reset",    None,         TEXT_DIM),
            ("[N]/[P] Config sau/trước", None, TEXT_DIM),
            ("[Q/Esc] Thoát",None,         TEXT_DIM),
        ]

        y = 10
        for item in lines:
            key_str, val_str, color = item
            if val_str is None:
                surf = self.font_l.render(key_str, True, color)
                self.screen.blit(surf, (px + 8, y))
            else:
                k_surf = self.font_l.render(key_str + ":", True, TEXT_DIM)
                v_surf = self.font_l.render(str(val_str), True, color)
                self.screen.blit(k_surf, (px + 8, y))
                self.screen.blit(v_surf, (px + PANEL_WIDTH - v_surf.get_width() - 8, y))
            y += 20
            if y > h - 30:
                break

        # Paused
        if self.paused:
            p_surf = self.font_h.render("⏸ PAUSED", True, (255, 200, 50))
            self.screen.blit(p_surf, (px + 8, h - 28))

        # Done
        if obs.get("done"):
            d_surf = self.font_h.render("✓ DONE", True, (46, 204, 113))
            self.screen.blit(d_surf, (px + 8, h - 28))

    # ------------------------------------------------------------------
    # Draw status bar
    # ------------------------------------------------------------------
    def draw_statusbar(self, obs: dict):
        w, h = self.screen.get_size()
        bar  = pygame.Rect(0, h - 30, w - PANEL_WIDTH - PADDING, 30)
        pygame.draw.rect(self.screen, (20, 20, 30), bar)
        t  = obs["t"]
        T  = obs["T"]
        pct = t / max(T, 1)
        prog_w = int((w - PANEL_WIDTH - PADDING * 2) * pct)
        pygame.draw.rect(self.screen, (52, 152, 219), (PADDING, h - 10, prog_w, 6))
        pygame.draw.rect(self.screen, (70, 70, 90),   (PADDING, h - 10, w - PANEL_WIDTH - PADDING * 3, 6), 1)
        lbl = self.font_l.render(f"t={t}/{T}  ({pct*100:.1f}%)", True, TEXT_DIM)
        self.screen.blit(lbl, (PADDING + 4, h - 28))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        running = True

        while running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_RIGHT:
                        self.speed = min(self.speed + 1, 30)
                    elif event.key == pygame.K_LEFT:
                        self.speed = max(1, self.speed - 1)
                    elif event.key == pygame.K_r:
                        self.stepping.reset()
                        self._last_step = time.time()
                    elif event.key == pygame.K_n:
                        self._switch_config(self.cfg_index + 1)
                    elif event.key == pygame.K_p:
                        self._switch_config(self.cfg_index - 1)

            # Advance simulation
            if not self.paused and not self.stepping.done:
                now = time.time()
                if now - self._last_step >= 1.0 / self.speed:
                    self.stepping.step()
                    self._last_step = now

            # Render
            self.screen.fill(BG_COLOR)
            obs      = self.stepping.obs
            grid     = obs["grid"]
            orders   = obs["orders"]
            shippers = obs["shippers"]

            cfg      = self.configs[self.cfg_index]

            self.draw_grid(grid)
            self.draw_orders(orders, shippers)
            self.draw_shippers(shippers, orders, self.stepping.next_actions)
            self.draw_panel(obs)
            self.draw_statusbar(obs)

            # Legend (góc trên-trái, trong grid)
            legend_items = [
                (PICKUP_COLOR,   "Pickup"),
                (DELIVERY_COLOR, "Delivery"),
                (CARRIED_COLOR,  "Đang mang"),
            ]
            lx, ly = self.grid_offset_x + 4, self.grid_offset_y + self.rows * self.cell + 4
            for color, label in legend_items:
                pygame.draw.rect(self.screen, color, (lx, ly, 12, 12))
                surf = self.font_s.render(label, True, TEXT_DIM)
                self.screen.blit(surf, (lx + 15, ly))
                lx += surf.get_width() + 35

            pygame.display.flip()

        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize DeliveryEnv")
    parser.add_argument("--config",       required=True,           help="Đường dẫn file test_config.txt")
    parser.add_argument("--method",       default="GreedyBFS",     help="Tên solver (mặc định: GreedyBFS)")
    parser.add_argument("--seed",         type=int, default=SEED,  help="Random seed")
    parser.add_argument("--speed",        type=int, default=5,     help="Bước/giây ban đầu (mặc định: 5)")
    parser.add_argument("--config-index", type=int, default=1,     help="Chỉ số config trong file (mặc định: 0)")
    args = parser.parse_args()

    print(f"Đọc config: {args.config}")
    configs = load_config(args.config)
    if not configs:
        sys.exit("[ERROR] Không tìm thấy config nào.")

    idx = args.config_index
    if idx < 1 or idx > len(configs):
        sys.exit(f"[ERROR] Không có config index={idx}. File có {len(configs)} config.")
    idx = idx - 1
    cfg = configs[idx]
    print(f"Dùng config: {cfg.get('name','?')}  N={cfg['N']} C={cfg['C']} G={cfg['G']} T={cfg['T']}")

    print(f"Load solver: {args.method}")
    solver_cls = get_solver_class(args.method)

    print("Khởi tạo xong. Mở cửa sổ visualizer...")
    renderer = Renderer(
        configs=configs,
        solver_cls=solver_cls,
        base_seed=args.seed,
        start_index=idx,
        speed=args.speed,
    )
    renderer.run()
    print("Đã thoát.")


if __name__ == "__main__":
    main()
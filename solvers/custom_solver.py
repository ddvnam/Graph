import time
from solvers.solver import Solver

class CustomSolver(Solver):
    def _decide_actions(self, obs) -> dict:
        return {shipper.id: ("S", 0) for shipper in obs["shippers"]}
    def run(self) -> dict:
        while not self.env.is_done():
            start = time.time()
            obs = self.env.reset()
            while not obs["done"]:
                actions = self._decide_actions(obs)
                obs, _, done, _ = self.env.step(actions)
                if done: break
            return self.env.result("CustomSolver", elapsed_sec=time.time() - start)
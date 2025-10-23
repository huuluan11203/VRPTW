import os
import json
import copy
import random
import math
from typing import List, Dict, Tuple
import numpy as np
from time import time
from collections import defaultdict, OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_processing import load_data_with_tw, compute_time_matrix, compute_time_matrix_OSRM, save_matrix

# =====================================================================
# CONFIGURATION
# =====================================================================
PENALTY_INFEASIBLE = 1e8
EPSILON = 1e-6 

# Cache configuration
MAX_CACHE_SIZE = 15000
CACHE_CLEAR_THRESHOLD = 0.85
OPERATOR_CACHE_SIZE = 5000

# Algorithm parameters
ELITE_POOL_SIZE = 10
T_START = 100.0
T_MIN = 0.01
COOLING_RATE = 0.99975


MAX_ITER = 50000
TIME_LIMIT = 60.0

# Weights & thresholds
SHAW_RELATEDNESS_PARAMS = {
    'distance_weight': 0.4,
    'time_weight': 0.3,
    'demand_weight': 0.2,
    'route_weight': 0.1
}

MAX_THREADS = os.cpu_count()  # số luồng đa luồng để tính toán song song


def _vehicle_penalty(excess: int) -> float:
    """Return penalty for using more vehicles than allowed (proportional)."""
    if excess <= 0:
        return 0.0
    return PENALTY_INFEASIBLE * excess

# =====================================================================
# ROUTE EVALUATION WITH MULTITHREADING
# =====================================================================
class RouteEvaluator:
    def __init__(self, data: Dict, distance_matrix: np.ndarray):
        self.data = data
        self.dist_matrix = distance_matrix
        # LRU cache for route evaluations (thread-safe)
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
        # Persistent executor to avoid creating it on every evaluation
        max_workers = min(MAX_THREADS, os.cpu_count() or MAX_THREADS)
        try:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        except Exception:
            # Fallback: single-threaded executor
            self.executor = ThreadPoolExecutor(max_workers=1)

    def evaluate(self, route: List[int]) -> Tuple[float, bool, Dict]:
        """Evaluate a single route with capacity constraint"""
        if not route:
            return 0.0, True, {'travel': 0, 'wait': 0, 'violations': [], 'total_demand': 0}

        route_key = tuple(route)
        # Try to read from cache with lock
        with self.cache_lock:
            if route_key in self.cache:
                # mark as recently used
                self.cache.move_to_end(route_key)
                self.cache_hits += 1
                return self.cache[route_key]

        self.cache_misses += 1
        depot = self.data["depot"]
        cust_map = self.data["customer_map"]
        capacity = self.data["capacity"]  

        current_time = depot["ready_time"]
        prev_node = 0
        total_travel = 0.0
        total_wait = 0.0
        total_demand = 0.0
        violations = []

        for node_idx in route:
            node_idx = int(node_idx)
            if node_idx not in cust_map:
                result = (PENALTY_INFEASIBLE, False, {'violations': ['invalid_node'], 'total_demand': 0})
                self.cache[route_key] = result
                return result

            cust = cust_map[node_idx]
            total_demand += cust["demand"]

            # Kiểm tra ràng buộc dung lượng
            if total_demand > capacity + EPSILON:
                violations.append(('capacity', node_idx, total_demand - capacity))

            travel = self.dist_matrix[prev_node, node_idx]
            if np.isinf(travel) or np.isnan(travel):
                result = (PENALTY_INFEASIBLE, False, {'violations': ['inf_distance'], 'total_demand': total_demand})
                self.cache[route_key] = result
                return result

            total_travel += travel
            current_time += travel

            ready = cust["ready_time"]
            due = cust["due_time"]

            if current_time < ready:
                wait = ready - current_time
                total_wait += wait
                current_time = ready

            if current_time > due + EPSILON:
                violations.append(('tw_late', node_idx, current_time - due))

            current_time += cust["service_time"]
            prev_node = node_idx

        travel_back = self.dist_matrix[prev_node, 0]
        if np.isinf(travel_back):
            result = (PENALTY_INFEASIBLE, False, {'violations': ['inf_return'], 'total_demand': total_demand})
            self.cache[route_key] = result
            return result

        total_travel += travel_back
        current_time += travel_back

        if current_time > depot["due_time"] + EPSILON:
            violations.append(('depot_late', 0, current_time - depot["due_time"]))

        is_feasible = len(violations) == 0
        cost = total_travel + 0.01 * total_wait if is_feasible else total_travel + sum(v[2] for v in violations) * 1000
        details = {'travel': total_travel, 'wait': total_wait, 'violations': violations, 'end_time': current_time, 'total_demand': total_demand}

        result = (cost, is_feasible, details)
        # Store in cache with lock and enforce LRU size
        with self.cache_lock:
            self.cache[route_key] = result
            # move to end done by assignment; enforce size
            if len(self.cache) > int(MAX_CACHE_SIZE * CACHE_CLEAR_THRESHOLD):
                # pop oldest until keep_size
                keep = int(MAX_CACHE_SIZE * (1.0 - CACHE_CLEAR_THRESHOLD))
                # ensure at least 1
                keep = max(1, keep)
                while len(self.cache) > keep:
                    try:
                        self.cache.popitem(last=False)
                    except Exception:
                        break
        return result

    def evaluate_solution(self, solution: List[List[int]]) -> Tuple[float, bool]:
        """Evaluate solution using multithreading"""
        total_cost = 0.0
        all_feasible = True

        # Check vehicle number constraint
        max_vehicles = self.data.get("vehicle_number", float('inf'))
        if len(solution) > max_vehicles:
            return PENALTY_INFEASIBLE * (len(solution) - max_vehicles), False

        # Reuse persistent executor to reduce overhead
        futures = {self.executor.submit(self.evaluate, route): route for route in solution}
        for future in as_completed(futures):
            cost, feasible, _ = future.result()
            total_cost += cost
            if not feasible:
                all_feasible = False

        return total_cost, all_feasible

    def clear_cache(self):
        with self.cache_lock:
            self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        # keep executor alive; do not shutdown here

    def shutdown(self):
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
# =====================================================================
# ADVANCED LOCAL SEARCH OPERATORS
# =====================================================================

def two_opt_star(solution: List[List[int]], evaluator: RouteEvaluator, 
                 max_attempts: int = 50) -> List[List[int]]:
    """
    2-opt* (inter-route): Exchange tails of two routes
    More powerful than standard 2-opt for multi-route problems
    """
    best_sol = copy.deepcopy(solution)
    best_cost, _ = evaluator.evaluate_solution(best_sol)
    # Local cache for route evaluations to avoid repeated evaluator calls
    route_costs = {}
    for idx, r in enumerate(best_sol):
        key = tuple(r)
        if key not in route_costs:
            route_costs[key] = evaluator.evaluate(r)
    improved = True
    attempts = 0
    
    while improved and attempts < max_attempts:
        improved = False
        attempts += 1
        
        # Try all pairs of routes
        for i in range(len(best_sol)):
            for j in range(i + 1, len(best_sol)):
                route_i = best_sol[i]
                route_j = best_sol[j]
                
                if not route_i or not route_j:
                    continue
                
                # Try all cut positions
                for pos_i in range(1, len(route_i)):
                    for pos_j in range(1, len(route_j)):
                        # Create new routes by swapping tails
                        new_route_i = route_i[:pos_i] + route_j[pos_j:]
                        new_route_j = route_j[:pos_j] + route_i[pos_i:]
                        
                        # Evaluate (use local cache when possible)
                        key_i = tuple(new_route_i)
                        key_j = tuple(new_route_j)
                        if key_i in route_costs:
                            cost_i, feas_i, _ = route_costs[key_i]
                        else:
                            cost_i, feas_i, _ = evaluator.evaluate(new_route_i)
                            route_costs[key_i] = (cost_i, feas_i, _)

                        if key_j in route_costs:
                            cost_j, feas_j, _ = route_costs[key_j]
                        else:
                            cost_j, feas_j, _ = evaluator.evaluate(new_route_j)
                            route_costs[key_j] = (cost_j, feas_j, _)
                        
                        if not (feas_i and feas_j):
                            continue
                        
                        # Calculate improvement (use cache)
                        old_key_i = tuple(route_i)
                        old_key_j = tuple(route_j)
                        old_cost_i, _, _ = route_costs.get(old_key_i, evaluator.evaluate(route_i))
                        route_costs.setdefault(old_key_i, (old_cost_i, True, {}))
                        old_cost_j, _, _ = route_costs.get(old_key_j, evaluator.evaluate(route_j))
                        route_costs.setdefault(old_key_j, (old_cost_j, True, {}))
                        
                        new_total = cost_i + cost_j
                        old_total = old_cost_i + old_cost_j
                        
                        if new_total < old_total - EPSILON:
                            best_sol[i] = new_route_i
                            best_sol[j] = new_route_j
                            best_cost = new_total - old_total + best_cost
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if improved:
                break
    
    return best_sol

def ejection_chain(solution: List[List[int]], evaluator: RouteEvaluator,
                   chain_length: int = 3) -> List[List[int]]:
    """
    Ejection chains: Sequence of moves where customers are ejected and reinserted
    Very powerful for VRPTW
    """
    best_sol = copy.deepcopy(solution)
    best_cost, _ = evaluator.evaluate_solution(best_sol)
    
    # Sample random starting customer
    all_customers = [c for route in best_sol for c in route]
    if not all_customers or len(all_customers) < chain_length:
        return best_sol
    
    seed = random.choice(all_customers)
    
    # Build ejection chain
    chain = [seed]
    current_sol = copy.deepcopy(best_sol)
    
    # Remove seed
    for route in current_sol:
        if seed in route:
            route.remove(seed)
            break
    
    # Try to build chain
    for _ in range(chain_length - 1):
        # Find best customer to eject next (closest to last in chain)
        candidates = [c for route in current_sol for c in route if c not in chain]
        if not candidates:
            break
        
        # Select based on distance
        last = chain[-1]
        distances = [(evaluator.dist_matrix[last, c], c) for c in candidates]
        distances.sort()
        
        # Eject closest
        next_customer = distances[0][1]
        chain.append(next_customer)
        
        for route in current_sol:
            if next_customer in route:
                route.remove(next_customer)
                break
    
    # Reinsert chain in best positions
    for customer in chain:
        best_insertion_cost = float('inf')
        best_route_idx = None
        best_position = None
        
        for ri, route in enumerate(current_sol):
            for pos in range(len(route) + 1):
                test_route = route[:pos] + [customer] + route[pos:]
                cost, feasible, _ = evaluator.evaluate(test_route)
                
                if feasible and cost < best_insertion_cost:
                    best_insertion_cost = cost
                    best_route_idx = ri
                    best_position = pos
        
        # Try new route
        new_route_cost, new_route_feas, _ = evaluator.evaluate([customer])
        if new_route_feas and new_route_cost < best_insertion_cost:
            current_sol.append([customer])
        elif best_route_idx is not None:
            current_sol[best_route_idx].insert(best_position, customer)
        else:
            # Forced insertion
            current_sol.append([customer])
    
    # Evaluate
    new_cost, feasible = evaluator.evaluate_solution(current_sol)
    if feasible and new_cost < best_cost:
        return current_sol
    
    return best_sol

def string_relocation(solution: List[List[int]], evaluator: RouteEvaluator,
                     max_string_len: int = 4) -> List[List[int]]:
    """
    String relocation: Move strings of customers between routes
    Extension of Or-opt to inter-route
    """
    best_sol = copy.deepcopy(solution)
    best_cost, _ = evaluator.evaluate_solution(best_sol)
    # Local cache for route evaluations
    route_costs = {}
    for idx, r in enumerate(best_sol):
        key = tuple(r)
        if key not in route_costs:
            route_costs[key] = evaluator.evaluate(r)
    improved = True
    
    while improved:
        improved = False
        
        for ri in range(len(best_sol)):
            route_i = best_sol[ri]
            if len(route_i) < 1:
                continue
            
            # Try different string lengths
            for str_len in range(1, min(max_string_len + 1, len(route_i) + 1)):
                for start_pos in range(len(route_i) - str_len + 1):
                    # Extract string
                    string = route_i[start_pos:start_pos + str_len]
                    route_i_without = route_i[:start_pos] + route_i[start_pos + str_len:]
                    
                    # Try inserting into other routes
                    for rj in range(len(best_sol)):
                        if ri == rj:
                            continue
                        
                        route_j = best_sol[rj]
                        
                        for insert_pos in range(len(route_j) + 1):
                            new_route_i = route_i_without
                            new_route_j = route_j[:insert_pos] + string + route_j[insert_pos:]
                            
                            # Evaluate (use local cache)
                            key_i = tuple(new_route_i)
                            key_j = tuple(new_route_j)
                            if key_i in route_costs:
                                cost_i, feas_i, _ = route_costs[key_i]
                            else:
                                cost_i, feas_i, _ = evaluator.evaluate(new_route_i)
                                route_costs[key_i] = (cost_i, feas_i, _)

                            if key_j in route_costs:
                                cost_j, feas_j, _ = route_costs[key_j]
                            else:
                                cost_j, feas_j, _ = evaluator.evaluate(new_route_j)
                                route_costs[key_j] = (cost_j, feas_j, _)
                            
                            if not (feas_i and feas_j):
                                continue
                            
                            old_key_i = tuple(route_i)
                            old_key_j = tuple(route_j)
                            old_cost_i, _, _ = route_costs.get(old_key_i, evaluator.evaluate(route_i))
                            route_costs.setdefault(old_key_i, (old_cost_i, True, {}))
                            old_cost_j, _, _ = route_costs.get(old_key_j, evaluator.evaluate(route_j))
                            route_costs.setdefault(old_key_j, (old_cost_j, True, {}))
                            
                            improvement = (old_cost_i + old_cost_j) - (cost_i + cost_j)
                            
                            if improvement > EPSILON:
                                best_sol[ri] = new_route_i
                                best_sol[rj] = new_route_j
                                best_cost -= improvement
                                improved = True
                                break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if improved:
                break
    
    return best_sol

# =====================================================================
# ADVANCED DESTROY OPERATORS
# =====================================================================

def shaw_removal_advanced(solution: List[List[int]], evaluator: RouteEvaluator,
                         q: int, params: Dict) -> Tuple[List[List[int]], List[int]]:
    """
    Advanced Shaw removal with multiple relatedness criteria
    """
    sol = copy.deepcopy(solution)
    all_customers = [c for route in sol for c in route]
    
    if not all_customers:
        return sol, []
    
    # Pick random seed
    seed = random.choice(all_customers)
    removed = [seed]
    
    # Calculate relatedness scores
    cust_map = evaluator.data["customer_map"]
    seed_data = cust_map[seed]
    
    relatedness = []
    for c in all_customers:
        if c == seed:
            continue
        
        c_data = cust_map[c]
        
        # Distance relatedness
        dist_rel = evaluator.dist_matrix[seed, c]
        dist_rel_norm = dist_rel / np.max(evaluator.dist_matrix)
        
        # Time window relatedness
        tw_diff = abs(seed_data["ready_time"] - c_data["ready_time"])
        tw_rel_norm = tw_diff / 1000.0  # Normalize
        
        # Demand relatedness
        demand_diff = abs(seed_data["demand"] - c_data["demand"])
        demand_rel_norm = demand_diff / 50.0  # Normalize
        
        # Route proximity (same route = more related)
        in_same_route = 0
        for route in sol:
            if seed in route and c in route:
                in_same_route = 1
                break
        route_rel = 1.0 - in_same_route
        
        # Combined relatedness (lower = more related)
        combined = (params['distance_weight'] * dist_rel_norm +
                   params['time_weight'] * tw_rel_norm +
                   params['demand_weight'] * demand_rel_norm +
                   params['route_weight'] * route_rel)
        
        relatedness.append((combined, c))
    
    # Sort and select most related
    relatedness.sort(key=lambda x: x[0])
    
    for i in range(min(q - 1, len(relatedness))):
        removed.append(relatedness[i][1])
    
    # Remove from solution
    for customer in removed:
        for route in sol:
            if customer in route:
                route.remove(customer)
                break
    
    return sol, removed

def time_oriented_removal(solution: List[List[int]], evaluator: RouteEvaluator,
                         q: int) -> Tuple[List[List[int]], List[int]]:
    """
    Remove customers with tight time windows or at critical times
    """
    sol = copy.deepcopy(solution)
    cust_map = evaluator.data["customer_map"]
    
    # Calculate time criticality for each customer
    criticalities = []
    for route in sol:
        for customer in route:
            c_data = cust_map[customer]
            tw_length = c_data["tw_length"]
            ready_time = c_data["ready_time"]
            
            # Tighter window = higher criticality
            criticality = 1.0 / (tw_length + 1)
            
            # Earlier time = higher priority (normalize)
            time_factor = 1.0 - (ready_time / 1500.0)
            
            combined = criticality + 0.3 * time_factor
            criticalities.append((combined, customer))
    
    if not criticalities:
        return sol, []
    
    # Sort by criticality (highest first)
    criticalities.sort(reverse=True, key=lambda x: x[0])
    
    # Remove top q
    removed = [c[1] for c in criticalities[:min(q, len(criticalities))]]
    
    for customer in removed:
        for route in sol:
            if customer in route:
                route.remove(customer)
                break
    
    return sol, removed

def cluster_removal(solution: List[List[int]], evaluator: RouteEvaluator,
                   q: int) -> Tuple[List[List[int]], List[int]]:
    """
    Remove spatially clustered customers
    """
    sol = copy.deepcopy(solution)
    all_customers = [c for route in sol for c in route]
    
    if not all_customers:
        return sol, []
    
    # Pick random center
    center = random.choice(all_customers)
    
    # Find q closest customers
    distances = []
    for c in all_customers:
        if c != center:
            dist = evaluator.dist_matrix[center, c]
            distances.append((dist, c))
    
    distances.sort(key=lambda x: x[0])
    
    removed = [center] + [d[1] for d in distances[:min(q - 1, len(distances))]]
    
    # Remove
    for customer in removed:
        for route in sol:
            if customer in route:
                route.remove(customer)
                break
    
    return sol, removed

# =====================================================================
# ADVANCED REPAIR OPERATORS
# =====================================================================


def regret_k_repair(solution: List[List[int]], removed: List[int],
                   evaluator: RouteEvaluator, k: int = 3) -> List[List[int]]:
    """
    Regret-k insertion with k=3 or higher
    """
    sol = copy.deepcopy(solution)
    if not sol:
        sol = [[]]
    
    remaining = set(removed)
    # Local cache for route evaluations to accelerate insertion scoring
    route_cache = {}
    
    while remaining:
        best_regret = -float('inf')
        best_customer = None
        best_route_idx = None
        best_position = None
        best_cost = None
        
        for customer in list(remaining):
            # Find k best insertions
            insertions = []
            
            for ri, route in enumerate(sol):
                # cache old route cost
                key_route = tuple(route)
                if key_route in route_cache:
                    old_cost = route_cache[key_route][0]
                else:
                    oc, ofeas, od = evaluator.evaluate(route)
                    route_cache[key_route] = (oc, ofeas, od)
                    old_cost = oc

                for pos in range(len(route) + 1):
                    test_route = route[:pos] + [customer] + route[pos:]
                    key_test = tuple(test_route)
                    if key_test in route_cache:
                        cost, feasible, _ = route_cache[key_test]
                    else:
                        cost, feasible, _ = evaluator.evaluate(test_route)
                        route_cache[key_test] = (cost, feasible, _)

                    if feasible:
                        insertion_cost = cost - old_cost
                        insertions.append((insertion_cost, ri, pos))
            
            # Try new route
            new_route_cost, new_feas, _ = evaluator.evaluate([customer])
            if new_feas:
                insertions.append((new_route_cost, len(sol), 0))
            
            if not insertions:
                continue
            
            # Sort insertions
            insertions.sort(key=lambda x: x[0])
            
            # Calculate regret-k
            if len(insertions) >= k:
                regret = sum(insertions[i][0] for i in range(1, k)) - (k - 1) * insertions[0][0]
            else:
                regret = sum(insertions[i][0] for i in range(1, len(insertions))) - (len(insertions) - 1) * insertions[0][0]
            
            if regret > best_regret:
                best_regret = regret
                best_customer = customer
                best_cost, best_route_idx, best_position = insertions[0]
        
        if best_customer is None:
            break
        
        # Insert
        if best_route_idx == len(sol):
            sol.append([best_customer])
        else:
            sol[best_route_idx].insert(best_position, best_customer)
        
        remaining.remove(best_customer)
    
    return sol

# =====================================================================
# ELITE SOLUTION POOL
# =====================================================================

class ElitePool:
    def __init__(self, size: int = 10):
        self.size = size
        self.solutions = []  # List of (cost, solution)
    
    def add(self, solution: List[List[int]], cost: float):
        """Add solution to pool if good enough"""
        # Check if solution already exists (diversity)
        sol_hash = self._hash_solution(solution)
        for _, existing_sol in self.solutions:
            if self._hash_solution(existing_sol) == sol_hash:
                return False
        
        # Add if better than worst or pool not full
        if len(self.solutions) < self.size:
            self.solutions.append((cost, copy.deepcopy(solution)))
            self.solutions.sort(key=lambda x: x[0])
            return True
        elif cost < self.solutions[-1][0]:
            self.solutions[-1] = (cost, copy.deepcopy(solution))
            self.solutions.sort(key=lambda x: x[0])
            return True
        
        return False
    
    def get_best(self) -> Tuple[float, List[List[int]]]:
        """Get best solution"""
        if self.solutions:
            return self.solutions[0]
        return float('inf'), []
    
    def get_random(self) -> List[List[int]]:
        """Get random solution from pool for diversification"""
        if self.solutions:
            _, sol = random.choice(self.solutions)
            return copy.deepcopy(sol)
        return []
    
    def _hash_solution(self, solution: List[List[int]]) -> int:
        """Simple hash for diversity check"""
        return hash(tuple(tuple(sorted(route)) for route in solution))

# =====================================================================
# MAIN ALNS ALGORITHM - ADVANCED
# =====================================================================

def alns_advanced(data: Dict, distance_matrix: np.ndarray,
                 max_iter: int,
                 time_limit: float,
                 alpha: float = 0.3,
                 removal_range: Tuple[int, int] = (5, 30),
                 adaptive: bool = False):
    """Adaptive Large Neighborhood Search - Advanced Version"""
    start_time = time()
    
    # Initialize evaluator
    evaluator = RouteEvaluator(data, distance_matrix)
    
    # Initial solution (better construction)
    solution = construct_initial_solution_advanced(data, evaluator)
    
    # Elite pool
    elite_pool = ElitePool(size=ELITE_POOL_SIZE)
    
    current_solution = copy.deepcopy(solution)
    current_cost, current_feas = evaluator.evaluate_solution(current_solution)
    
    best_solution = copy.deepcopy(current_solution)
    best_cost = current_cost
    
    elite_pool.add(best_solution, best_cost)
    
    print(f"Initial: {len(solution)} routes, cost={best_cost:.2f}, feasible={current_feas}")
    
    # Setup operators
    destroy_ops = [
        ("shaw_advanced", lambda s, q: shaw_removal_advanced(s, evaluator, q, SHAW_RELATEDNESS_PARAMS)),
        ("time_oriented", lambda s, q: time_oriented_removal(s, evaluator, q)),
        ("cluster", lambda s, q: cluster_removal(s, evaluator, q)),
        ("random", lambda s, q: random_removal(s, q)),
        ("worst", lambda s, q: worst_removal(s, evaluator, q))
    ]
    
    repair_ops = [
        ("regret3", lambda s, r: regret_k_repair(s, r, evaluator, k=3)),
        ("regret4", lambda s, r: regret_k_repair(s, r, evaluator, k=4))
    ]
    
    local_search_ops = [
        ("2opt_star", lambda s: two_opt_star(s, evaluator)),
        ("ejection", lambda s: ejection_chain(s, evaluator, chain_length=3)),
        ("string_reloc", lambda s: string_relocation(s, evaluator, max_string_len=3))
    ]
    
    # UCB statistics
    destroy_stats = {name: {"count": 1, "reward": 0.1} for name, _ in destroy_ops}
    repair_stats = {name: {"count": 1, "reward": 0.1} for name, _ in repair_ops}
    
    # SA parameters (can be adapted by AdaptiveParameters)
    T = max(50.0, best_cost * 0.1 if best_cost and best_cost > 0 else 500.0)
    T_min = 0.1
    cooling = 0.9995

    params = AdaptiveParameters() if adaptive else None
    if params is not None:
        params.T = T
    
    no_improve_count = 0
    iteration = 0
    
    print(f"\nRunning ALNS (max_iter={max_iter}, time_limit={time_limit}s, max_threads = {MAX_THREADS})\n")
    
    # Main loop
    while iteration < max_iter:
        iteration += 1
        elapsed = time() - start_time
        
        if elapsed > time_limit:
            print(f"\n⏰ Time limit reached at iteration {iteration}")
            break
        
        # Select operators using UCB
        d_name, d_func = select_ucb(destroy_ops, destroy_stats, alpha, iteration)
        r_name, r_func = select_ucb(repair_ops, repair_stats, alpha, iteration)
        
        # Removal size
        q = random.randint(removal_range[0], removal_range[1])
        q = min(q, sum(len(r) for r in current_solution))
        
        # Apply destroy-repair
        partial, removed = d_func(current_solution, q)
        candidate = r_func(partial, removed)
        
        # Remove empty routes
        candidate = [r for r in candidate if r]
        
        # Apply local search (selectively)
        if iteration % 5 == 0:
            ls_name, ls_func = random.choice(local_search_ops)
            candidate = ls_func(candidate)
        
        # Evaluate
        cand_cost, cand_feas = evaluator.evaluate_solution(candidate)
        
        # Acceptance
        delta = cand_cost - current_cost
        accept = False
        improved = False

        if delta < -EPSILON:
            accept = True
            improved = True
        else:
            # use adaptive temperature if enabled
            temp = params.T if params is not None else T
            if temp > T_min:
                prob = math.exp(-delta / temp)
                if random.random() < prob:
                    accept = True
        
        # Update
        reward = 0.0
        if accept:
            current_solution = candidate
            current_cost = cand_cost
            
            if cand_cost < best_cost - EPSILON:
                improvement = best_cost - cand_cost
                best_solution = copy.deepcopy(candidate)
                best_cost = cand_cost
                elite_pool.add(best_solution, best_cost)
                reward = min(100.0, improvement)
                no_improve_count = 0
                
                print(f"  [{iteration:5d}] - BEST: {best_cost:.2f} "
                      f"(Δ={improvement:.2f}, routes={len(best_solution)}, "
                      f"d={d_name}, r={r_name})")
            else:
                reward = 0.1
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Update UCB
        destroy_stats[d_name]["count"] += 1
        destroy_stats[d_name]["reward"] += reward
        repair_stats[r_name]["count"] += 1
        repair_stats[r_name]["reward"] += reward
        
        # Cooling / adaptive updates
        if params is not None:
            params.update(accept, improved)
            # sync T and removal_range
            T = params.T
            removal_range = params.get_removal_range()
        else:
            T = max(T_min, T * cooling)
        
        # Diversification strategies
        if no_improve_count >= 150:
            print(f"  [{iteration:5d}] Diversification (no improvement for {no_improve_count} iters)")
            
            # Strategy 1: Jump to elite solution
            if random.random() < 0.5 and len(elite_pool.solutions) > 1:
                current_solution = elite_pool.get_random()
                current_cost, _ = evaluator.evaluate_solution(current_solution)
            # Strategy 2: Large perturbation
            else:
                large_q = min(50, int(0.4 * sum(len(r) for r in current_solution)))
                partial, removed = shaw_removal_advanced(best_solution, evaluator, large_q, SHAW_RELATEDNESS_PARAMS)
                current_solution = regret_k_repair(partial, removed, evaluator, k=4)
                current_cost, _ = evaluator.evaluate_solution(current_solution)
            
            no_improve_count = 0
            T = 500.0  # Reheat
        
        # Adaptive removal size
        if iteration % 100 == 0:
            if no_improve_count > 50:
                # Increase removal size for more diversification
                removal_range = (removal_range[0], min(50, removal_range[1] + 5))
            else:
                # Reset to normal
                removal_range = (5, 30)
        
        # Progress report
        if iteration % 100 == 0:
            print(f"  [{iteration:5d}] current={current_cost:.2f}, best={best_cost:.2f}, "
                  f"T={T:.2f}, elapsed={elapsed:.1f}s")
            print(f"           cache: {evaluator.cache_hits} hits, {evaluator.cache_misses} misses")
    
    # Post-optimization
    print(f"\n🔧 Post-optimization...")
    best_solution = post_optimize(best_solution, evaluator, time_limit=30)
    best_cost, best_feas = evaluator.evaluate_solution(best_solution)
    
    # Final report
    elapsed = time() - start_time
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Best cost: {best_cost:.2f}")
    print(f"Routes: {len(best_solution)}")
    print(f"Feasible: {best_feas}")
    print(f"Iterations: {iteration}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Cache efficiency: {evaluator.cache_hits/(evaluator.cache_hits + evaluator.cache_misses)*100:.1f}%")
    
    return best_solution, best_cost

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def select_ucb(operators: List[Tuple[str, callable]], stats: Dict, 
               alpha: float, iteration: int) -> Tuple[str, callable]:
    """UCB selection with adaptive alpha and operator history"""
    scores = {}
    total_rewards = sum(s["reward"] for s in stats.values())
    total_count = sum(s["count"] for s in stats.values())
    
    for name, func in operators:
        # Cơ bản UCB
        avg_reward = stats[name]["reward"] / stats[name]["count"]
        exploration = math.sqrt(alpha * math.log(iteration + 1) / stats[name]["count"])
        
        # Thêm yếu tố thành công gần đây
        recent_success = stats[name].get("recent_success", 0)
        recency_factor = 0.3 * recent_success / max(1, stats[name].get("recent_count", 1))
        
        # Thêm yếu tố đa dạng
        diversity_factor = 0.2 * (1.0 - stats[name]["count"] / total_count)
        
        # Kết hợp các yếu tố
        scores[name] = avg_reward + exploration + recency_factor + diversity_factor
    
    # Select best
    best_name = max(scores, key=scores.get)
    
    # Find the function
    for name, func in operators:
        if name == best_name:
            return name, func
    
    return operators[0]

def random_removal(solution: List[List[int]], q: int) -> Tuple[List[List[int]], List[int]]:
    """Simple random removal"""
    sol = copy.deepcopy(solution)
    all_customers = [(ri, idx, node) for ri, r in enumerate(sol) for idx, node in enumerate(r)]
    
    if not all_customers:
        return sol, []
    
    q = min(q, len(all_customers))
    removed = []
    
    for _ in range(q):
        if not all_customers:
            break
        ri, idx, node = random.choice(all_customers)
        removed.append(node)
        sol[ri].pop(idx)
        all_customers = [(rj, idj, nd) for rj, r in enumerate(sol) for idj, nd in enumerate(r)]
    
    return sol, removed

def worst_removal(solution: List[List[int]], evaluator: RouteEvaluator,
                  q: int) -> Tuple[List[List[int]], List[int]]:
    """Remove worst customers by cost contribution"""
    sol = copy.deepcopy(solution)
    deltas = []
    # Local cache for route evaluations
    route_cache = {}
    for ri, r in enumerate(sol):
        key_r = tuple(r)
        if key_r in route_cache:
            before, _, _ = route_cache[key_r]
        else:
            before = evaluator.evaluate(r)
            # store only cost,feas,details tuple
            route_cache[key_r] = before
            before = before[0]

        for idx, node in enumerate(r):
            r_without = r[:idx] + r[idx + 1:]
            if not r_without:
                after_cost = 0.0
            else:
                key_wo = tuple(r_without)
                if key_wo in route_cache:
                    after_cost = route_cache[key_wo][0]
                else:
                    after = evaluator.evaluate(r_without)
                    route_cache[key_wo] = after
                    after_cost = after[0]

            saving = before - after_cost
            deltas.append((saving, ri, idx, node))
    
    if not deltas:
        return sol, []
    
    deltas.sort(reverse=True, key=lambda x: x[0])
    
    removed = []
    for k in range(min(q, len(deltas))):
        _, ri, _, node = deltas[k]
        if node in sol[ri]:
            sol[ri].remove(node)
            removed.append(node)
    
    return sol, removed

def construct_initial_solution_advanced(data: Dict, evaluator: RouteEvaluator) -> List[List[int]]:
    """
    Advanced initial solution construction
    Uses time-oriented nearest neighbor with regret
    """
    cust_indices = [c["idx"] for c in data["customers"]]
    unserved = set(cust_indices)
    solution = []
    
    # Sort customers by ready time
    cust_by_time = sorted(cust_indices, key=lambda c: data["customer_map"][c]["ready_time"])
    
    while unserved:
        route = []
        current_node = 0
        current_time = data["depot"]["ready_time"]
        
        # Build route using time-oriented nearest neighbor
        while unserved:
            best_customer = None
            best_score = float('inf')
            
            for c in list(unserved):
                c_data = data["customer_map"][c]
                
                # Calculate arrival time
                travel = evaluator.dist_matrix[current_node, c]
                arrival = current_time + travel
                
                # Check time window
                if arrival > c_data["due_time"]:
                    continue
                
                # Wait if early
                start_service = max(arrival, c_data["ready_time"])
                
                # Score: distance + time window slack + urgency
                tw_slack = c_data["due_time"] - start_service
                urgency = 1.0 / (tw_slack + 1)  # Higher for tight windows
                
                score = travel + 50 * urgency
                
                if score < best_score:
                    best_score = score
                    best_customer = c
            
            if best_customer is None:
                break
            
            # Add to route
            route.append(best_customer)
            unserved.remove(best_customer)
            
            c_data = data["customer_map"][best_customer]
            travel = evaluator.dist_matrix[current_node, best_customer]
            arrival = current_time + travel
            start_service = max(arrival, c_data["ready_time"])
            current_time = start_service + c_data["service_time"]
            current_node = best_customer
        
        if route:
            solution.append(route)
    
    return solution

def post_optimize(solution: List[List[int]], evaluator: RouteEvaluator,
                 time_limit: float = 30) -> List[List[int]]:
    """
    Post-optimization with intensive local search và nhiều chiến lược tối ưu
    """
    best_sol = copy.deepcopy(solution)
    best_cost, _ = evaluator.evaluate_solution(best_sol)
    
    start = time()
    iteration = 0
    no_improve = 0
    phase = 0  # Pha tối ưu
    
    while time() - start < time_limit and no_improve < 50:
        iteration += 1
        improved = False
        
        # Luân phiên các chiến lược tối ưu
        if phase == 0:
            # Tối ưu trong tuyến
            for i in range(len(best_sol)):
                new_route = or_opt_intensive(best_sol[i], evaluator, max_len=4)
                new_cost, _, _ = evaluator.evaluate(new_route)
                old_cost, _, _ = evaluator.evaluate(best_sol[i])
                if new_cost < old_cost:
                    best_sol[i] = new_route
                    improved = True
        
        elif phase == 1:
            # Tối ưu giữa các tuyến
            new_sol = two_opt_star(best_sol, evaluator, max_attempts=20)
            new_cost, _ = evaluator.evaluate_solution(new_sol)
            if new_cost < best_cost:
                best_sol = new_sol
                best_cost = new_cost
                improved = True
        
        elif phase == 2:
            # Tối ưu chuỗi khách hàng
            new_sol = string_relocation(best_sol, evaluator, max_string_len=3)
            new_cost, _ = evaluator.evaluate_solution(new_sol)
            if new_cost < best_cost:
                best_sol = new_sol
                best_cost = new_cost
                improved = True
        
        elif phase == 3:
            # Ejection chains chuyên sâu
            new_sol = ejection_chain(best_sol, evaluator, chain_length=4)
            new_cost, _ = evaluator.evaluate_solution(new_sol)
            if new_cost < best_cost:
                best_sol = new_sol
                best_cost = new_cost
                improved = True
        
        # Intra-route: Or-opt on each route
        for i in range(len(best_sol)):
            if not best_sol[i]:
                continue
            
            old_route = best_sol[i]
            new_route = or_opt_intensive(old_route, evaluator)
            
            if new_route != old_route:
                old_cost, _, _ = evaluator.evaluate(old_route)
                new_cost, _, _ = evaluator.evaluate(new_route)
                
                if new_cost < old_cost - EPSILON:
                    best_sol[i] = new_route
                    improved = True
        
        # Inter-route: 2-opt*
        best_sol = two_opt_star(best_sol, evaluator, max_attempts=10)
        
        # String relocation
        best_sol = string_relocation(best_sol, evaluator, max_string_len=2)
        
        if not improved:
            break
    
    return best_sol


class AdaptiveParameters:
    """Advanced adaptive parameters manager with multi-criteria adaptation"""
    def __init__(self):
        # Temperature parameters
        self.T = T_START
        self.T_min = T_MIN
        self.cooling = COOLING_RATE
        self.reheat_factor = 2.0
        
        # Removal range
        self.removal_min = 4
        self.removal_max = 35
        
        # Operator weights
        self.destroy_weights = defaultdict(lambda: 1.0)
        self.repair_weights = defaultdict(lambda: 1.0)
        self.weight_adjustment = 0.1
        
        # History tracking
        self.accept_history = []
        self.improve_history = []
        self.window_size = 100
        self.stagnation_count = 0
        
        # Performance metrics
        self.iterations_since_improve = 0
        self.total_accepted = 0
        self.total_improved = 0

    def update(self, accepted: bool, improved: bool):
        self.accept_history.append(1 if accepted else 0)
        self.improve_history.append(1 if improved else 0)
        window = 100
        if len(self.accept_history) > window:
            self.accept_history = self.accept_history[-window:]
            self.improve_history = self.improve_history[-window:]

        accept_rate = sum(self.accept_history) / max(1, len(self.accept_history))
        improve_rate = sum(self.improve_history) / max(1, len(self.improve_history))

        if accept_rate < 0.05:
            self.cooling = min(0.9999, self.cooling * 1.0005)
        elif accept_rate > 0.5:
            self.cooling = max(0.995, self.cooling * 0.999)

        if improve_rate < 0.02:
            self.removal_max = min(60, self.removal_max + 2)
        elif improve_rate > 0.1:
            self.removal_max = max(20, self.removal_max - 2)


# =====================================================================
# DELTA EVALUATION HELPERS
# =====================================================================



def compute_delta_string_relocation(from_route: List[int], to_route: List[int],
                                  string_start: int, string_len: int, insert_pos: int,
                                  evaluator: RouteEvaluator) -> Tuple[float, bool]:
    """Delta evaluator for string relocation moves.
    Returns (delta_distance, feasible_estimate).
    Conservative estimate - checks distance and capacity.
    """
    if string_start < 0 or string_start + string_len > len(from_route):
        return float('inf'), False
    if insert_pos < 0 or insert_pos > len(to_route):
        return float('inf'), False
        
    # Get string to move
    string = from_route[string_start:string_start + string_len]
    
    # Quick capacity check for target route
    cust_map = evaluator.data['customer_map']
    string_demand = sum(cust_map[c]['demand'] for c in string)
    target_demand = sum(cust_map[c]['demand'] for c in to_route)
    
    if target_demand + string_demand > evaluator.data['capacity'] + EPSILON:
        return float('inf'), False
    
    # Calculate distance changes in source route
    source_before = from_route[string_start - 1] if string_start > 0 else 0
    source_after = from_route[string_start + string_len] if string_start + string_len < len(from_route) else 0
    
    old_source = (evaluator.dist_matrix[source_before, string[0]] +
                 evaluator.dist_matrix[string[-1], source_after])
    new_source = evaluator.dist_matrix[source_before, source_after]
    
    # Calculate distance changes in target route
    target_before = to_route[insert_pos - 1] if insert_pos > 0 else 0
    target_after = to_route[insert_pos] if insert_pos < len(to_route) else 0
    
    old_target = evaluator.dist_matrix[target_before, target_after]
    new_target = (evaluator.dist_matrix[target_before, string[0]] +
                 evaluator.dist_matrix[string[-1], target_after])
    
    delta = (new_source - old_source) + (new_target - old_target)
    return delta, True

def compute_delta_oropt(route: List[int], i: int, length: int, k: int,
                        evaluator: RouteEvaluator) -> Tuple[float, bool]:
    """Fast conservative delta estimator for Or-opt moves.
    Returns (delta_distance, feasible_estimate). This function only
    checks distance and capacity quickly; it does not fully validate
    time windows (conservative True means 'maybe feasible').
    Use as pre-filter: if delta is large or capacity violated, skip full eval.
    """
    n = len(route)
    if i < 0 or i + length > n or k < 0 or k > n - length + 1:
        return float('inf'), False

    subseq = route[i:i+length]
    before = route[i-1] if i > 0 else 0
    after = route[i+length] if (i+length) < n else 0

    # insertion point in remaining: position k in remaining corresponds to index in new route
    # We map remaining indices to actual nodes for quick distance delta
    remaining = route[:i] + route[i+length:]
    if k < 0 or k > len(remaining):
        return float('inf'), False
    before_k = remaining[k-1] if k > 0 else 0
    after_k = remaining[k] if k < len(remaining) else 0

    # old edges removed
    old = evaluator.dist_matrix[before, route[i]] + evaluator.dist_matrix[route[i+length-1], after]
    if before_k is not None and after_k is not None and (k > i):
        # if insertion is to the right of original block, original before_k->after_k edge exists
        old += evaluator.dist_matrix[before_k, after_k]

    # new edges added
    new = evaluator.dist_matrix[before, after] + evaluator.dist_matrix[before_k, route[i]] + evaluator.dist_matrix[route[i+length-1], after_k]

    delta = new - old

    # Quick capacity check: compute demand of subseq and remaining prefix/suffix
    cust_map = evaluator.data['customer_map']
    subseq_demand = sum(cust_map[c]['demand'] for c in subseq)
    # compute route demand prefix/suffix quickly
    total_demand = sum(cust_map[c]['demand'] for c in route)
    capacity = evaluator.data.get('capacity', float('inf'))
    if subseq_demand > capacity + EPSILON:
        return float('inf'), False

    # Conservative TW feasibility: return True (maybe feasible) — full check later
    return delta, True

def or_opt_intensive(route: List[int], evaluator: RouteEvaluator,
                     max_len: int = 3) -> List[int]:
    """Intensive Or-opt with all lengths"""
    if len(route) <= 2:
        return route

    best = route[:]
    # Local cache for route evaluations inside this intensive routine
    route_cache = {}
    best_cost, _, _ = evaluator.evaluate(best)
    route_cache[tuple(best)] = (best_cost, True, {})
    improved = True
    
    while improved:
        improved = False
        
        for length in range(1, min(max_len + 1, len(best))):
            for i in range(len(best) - length + 1):
                subseq = best[i:i + length]
                remaining = best[:i] + best[i + length:]
                
                for k in range(len(remaining) + 1):
                    if k == i:
                        continue
                    # delta pre-check to skip many full evaluations
                    delta, maybe_feas = compute_delta_oropt(best, i, length, k, evaluator)
                    if not maybe_feas:
                        continue
                    # quick filter: if delta is not promising, skip
                    if delta > 1.0:  # threshold, keep conservative
                        continue

                    new_route = remaining[:k] + subseq + remaining[k:]
                    key_new = tuple(new_route)
                    if key_new in route_cache:
                        cost, feas, _ = route_cache[key_new]
                    else:
                        cost, feas, _ = evaluator.evaluate(new_route)
                        route_cache[key_new] = (cost, feas, _)

                    if feas and cost < best_cost - EPSILON:
                        best = new_route
                        best_cost = cost
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
    
    return best

# =====================================================================
# MAIN RUNNER
# =====================================================================

def run_vrptw_advanced(customers_file: str,
                      max_iter: int,
                      time_limit: float,
                      use_cache: bool = True):
    """Advanced VRPTW solver with multithreading"""

    import os

    HERE = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(HERE)
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    results_dir = os.path.join(PROJECT_ROOT, "results")

    #path
    results_path = os.path.join(results_dir, "solution_advanced.json")
    cache_path = os.path.join(processed_dir, "distance_matrix_cache.json")

    #remove all data files
    try:
        os.remove(cache_path)
        os.remove(results_path)
        print("Đã xóa các file dữ liệu cũ.")
    except OSError as e:
        print(f"Không thể xóa file gốc: {e}")

    data, locations = load_data_with_tw(customers_file)

    # Distance matrix
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            print("Loading cached distance matrix")
            matrix_list = json.load(f)
        # matrix_list may contain nulls for unreachable entries; convert to np.inf
        distance_matrix = np.array([[ (float(v) if v is not None else float('inf')) for v in row] for row in matrix_list], dtype=float)
    else:
        if len(locations) > 99:
            matrix_list = compute_time_matrix(locations)
        else:
            matrix_list = compute_time_matrix_OSRM(locations)
        # Save matrix in minutes (and normalize invalid entries to null in JSON)
        save_matrix(matrix_list, filename=os.path.basename(cache_path), to_minutes=True)
        # load back saved JSON to ensure same format
        with open(cache_path, "r") as f:
            matrix_list_saved = json.load(f)
        distance_matrix = np.array([[ (float(v) if v is not None else float('inf')) for v in row] for row in matrix_list_saved], dtype=float)
    # distance_matrix is already in minutes (saved that way by save_matrix). Ensure numeric and inf handled.

    best_solution, best_cost = alns_advanced(
        data=data,
        distance_matrix=distance_matrix,
        max_iter=max_iter,
        time_limit=time_limit,
        alpha=0.3,                          
        removal_range=(5, 30)
    )

    idx_to_id = data["idx_to_id"]
    best_solution_orig = [[idx_to_id[idx] for idx in route] for route in best_solution]

    # Save solution
    output_file = "solution_advanced.json"
    output_path = os.path.join(results_dir, output_file)
    solution_data = {"cost": best_cost, "num_routes": len(best_solution_orig), "routes": best_solution_orig, "route_details": []}
    evaluator = RouteEvaluator(data, distance_matrix)
    for i, route in enumerate(best_solution):
        cost, feasible, details = evaluator.evaluate(route)
        solution_data["route_details"].append({
            "route_num": i + 1,
            "customers": [idx_to_id[idx] for idx in route],
            "cost": cost,
            "travel_time": details.get('travel', 0),
            "wait_time": details.get('wait', 0),
            "feasible": feasible
        })
    with open(output_path, "w") as f:
        json.dump(solution_data, f, indent=2)

    # Clean up persistent executor(s)
    try:
        evaluator.shutdown()
    except Exception:
        pass

    return best_solution, best_cost

# =====================================================================
# CLI
# =====================================================================

def main(input_file: str):
    run_vrptw_advanced(
        customers_file=input_file,
        max_iter=MAX_ITER,
        time_limit=TIME_LIMIT,
        use_cache=True,
    )



if __name__ == "__main__":
    import sys

    input_file = "test90.csv"

    best_sol, best_cost = run_vrptw_advanced(
        customers_file=input_file,
        max_iter=MAX_ITER,
        time_limit=TIME_LIMIT,
        use_cache=True,
    )
#pragma once

#include "types.hpp"
#include "dag.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <memory>

namespace agentic_or {

class RCPSPSolver {
public:
    TaskDAG dag;
    ResourceLimits limits;
    double weight_makespan{1.0};
    double weight_setup{0.2};
    double weight_resource_stress{0.5};

    RCPSPSolver(const TaskDAG& dag, const ResourceLimits& limits)
        : dag(dag), limits(limits), rng_(42) {}

    void set_weights(double alpha, double beta, double gamma) {
        weight_makespan = alpha;
        weight_setup = beta;
        weight_resource_stress = gamma;
    }

    ScheduleResult solve(int time_budget_ms = 150);

private:
    std::mt19937 rng_;

    // Internal solution representation
    struct Solution {
        std::vector<int> worker_of_task;       // task_index -> worker_id (-1 if unassigned)
        std::vector<int> start_of_task;        // task_index -> start_time_ms
        std::vector<int> end_of_task;          // task_index -> end_time_ms
        std::vector<int> setup_of_task;        // task_index -> setup_cost_ms
        std::vector<std::vector<int>> worker_schedule; // worker_id -> ordered list of task indices
        
        int makespan{0};
        int total_setup_cost{0};
        double score{1e9};
        bool is_valid{false};
    };

    int calculate_setup_cost(int prev_task_idx, int next_task_idx) const;
    void evaluate_solution(Solution& sol) const;
    Solution construct_initial_greedy_solution();

    // Destroy operators (return unassigned task indices)
    std::vector<int> destroy_critical_path(Solution& sol, const TaskDAG::CPMResult& cpm, double fraction);
    std::vector<int> destroy_resource_peak(Solution& sol, double fraction);
    std::vector<int> destroy_worst_setup(Solution& sol, double fraction);
    std::vector<int> destroy_domain_cluster(Solution& sol);

    // Repair operators (re-insert unassigned tasks)
    void repair_context_affinity(Solution& sol, std::vector<int>& unassigned);
    void repair_regret_k(Solution& sol, std::vector<int>& unassigned, int k = 2);
    void repair_earliest_feasible(Solution& sol, std::vector<int>& unassigned);
    void repair_priority_slack(Solution& sol, std::vector<int>& unassigned, const TaskDAG::CPMResult& cpm);

    void recompute_worker_schedule(Solution& sol, int worker_id) const;
    void recompute_all_schedules(Solution& sol) const;

    bool find_best_insertion(
        const Solution& sol,
        int task_idx,
        int& best_worker,
        int& best_pos,
        int& best_start,
        int& best_setup,
        double& best_delta_cost
    ) const;

    void insert_task_into_worker(Solution& sol, int task_idx, int worker_id, int pos, int start_time, int setup_cost);
    void remove_task_from_solution(Solution& sol, int task_idx);
};

} // namespace agentic_or

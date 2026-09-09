#include "alns.hpp"
#include <map>
#include <cmath>
#include <numeric>
#include <climits>

namespace agentic_or {

int RCPSPSolver::calculate_setup_cost(int prev_task_idx, int next_task_idx) const {
    const auto& next_task = dag.tasks[next_task_idx];
    if (prev_task_idx < 0) {
        // Cold start on an empty worker
        if (next_task.workload_type == WorkloadType::BROWSER) {
            return limits.browser_cold_start_ms;
        }
        return limits.local_setup_ms;
    }

    const auto& prev_task = dag.tasks[prev_task_idx];
    if (next_task.workload_type == WorkloadType::BROWSER) {
        if (prev_task.workload_type == WorkloadType::BROWSER) {
            // Both are browser tasks
            if (!next_task.affinity_key.empty() && next_task.affinity_key == prev_task.affinity_key) {
                // Same domain / session: zero cold start!
                return 0;
            }
            // Different domain: new tab in warm browser
            return limits.browser_tab_reuse_ms;
        } else {
            // Previous was non-browser, now launching browser
            return limits.browser_cold_start_ms;
        }
    }

    return limits.local_setup_ms;
}

void RCPSPSolver::evaluate_solution(Solution& sol) const {
    int max_end = 0;
    int total_setup = 0;
    double resource_penalty = 0.0;

    // Check precedence and worker integrity
    bool valid = true;
    for (size_t i = 0; i < dag.tasks.size(); ++i) {
        if (sol.worker_of_task[i] < 0) {
            valid = false;
            break;
        }
        max_end = std::max(max_end, sol.end_of_task[i]);
        total_setup += sol.setup_of_task[i];

        // Verify precedence
        for (int p : dag.tasks[i].predecessors) {
            if (sol.start_of_task[i] < sol.end_of_task[p]) {
                valid = false;
                break;
            }
        }
    }

    // Measure resource stress: find peak RAM usage
    if (valid && max_end > 0) {
        // Discretized sweep or event-based sweep of RAM usage
        struct Event {
            int time;
            int ram_delta;
            double cpu_delta;
        };
        std::vector<Event> events;
        events.reserve(dag.tasks.size() * 2);
        for (size_t i = 0; i < dag.tasks.size(); ++i) {
            events.push_back({sol.start_of_task[i], dag.tasks[i].ram_mb, dag.tasks[i].cpu_percent});
            events.push_back({sol.end_of_task[i], -dag.tasks[i].ram_mb, -dag.tasks[i].cpu_percent});
        }
        std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
            if (a.time == b.time) return a.ram_delta < b.ram_delta;
            return a.time < b.time;
        });

        int curr_ram = 0;
        int peak_ram = 0;
        double curr_cpu = 0.0;
        double peak_cpu = 0.0;
        for (const auto& ev : events) {
            curr_ram += ev.ram_delta;
            curr_cpu += ev.cpu_delta;
            peak_ram = std::max(peak_ram, curr_ram);
            peak_cpu = std::max(peak_cpu, curr_cpu);
        }

        if (peak_ram > limits.max_ram_mb) {
            resource_penalty += (peak_ram - limits.max_ram_mb) * 10.0;
        }
        if (peak_cpu > limits.max_cpu_percent) {
            resource_penalty += (peak_cpu - limits.max_cpu_percent) * 50.0;
        }
    }

    sol.makespan = max_end;
    sol.total_setup_cost = total_setup;
    sol.is_valid = valid;
    sol.score = (weight_makespan * max_end) +
                (weight_setup * total_setup) +
                (weight_resource_stress * resource_penalty);
}

void RCPSPSolver::recompute_worker_schedule(Solution& sol, int w) const {
    if (w < 0 || w >= static_cast<int>(sol.worker_schedule.size())) return;
    auto& sched = sol.worker_schedule[w];
    int current_time = 0;
    int prev_task = -1;
    for (int t : sched) {
        int setup = calculate_setup_cost(prev_task, t);
        sol.setup_of_task[t] = setup;

        int pred_earliest = 0;
        for (int pred : dag.tasks[t].predecessors) {
            pred_earliest = std::max(pred_earliest, sol.end_of_task[pred]);
        }

        int start = std::max(pred_earliest, current_time + setup);
        sol.start_of_task[t] = start;
        sol.end_of_task[t] = start + dag.tasks[t].duration_ms;
        current_time = sol.end_of_task[t];
        prev_task = t;
    }
}

void RCPSPSolver::recompute_all_schedules(Solution& sol) const {
    std::vector<int> topo = dag.topological_sort();
    for (int w = 0; w < static_cast<int>(sol.worker_schedule.size()); ++w) {
        recompute_worker_schedule(sol, w);
    }
}

void RCPSPSolver::remove_task_from_solution(Solution& sol, int task_idx) {
    int w = sol.worker_of_task[task_idx];
    if (w < 0) return;

    auto& sched = sol.worker_schedule[w];
    auto it = std::find(sched.begin(), sched.end(), task_idx);
    if (it != sched.end()) {
        sched.erase(it);
    }

    sol.worker_of_task[task_idx] = -1;
    sol.start_of_task[task_idx] = 0;
    sol.end_of_task[task_idx] = 0;
    sol.setup_of_task[task_idx] = 0;
    recompute_worker_schedule(sol, w);
}

void RCPSPSolver::insert_task_into_worker(
    Solution& sol, int task_idx, int worker_id, int pos, int start_time, int setup_cost
) {
    auto& sched = sol.worker_schedule[worker_id];
    if (pos >= static_cast<int>(sched.size())) {
        sched.push_back(task_idx);
    } else {
        sched.insert(sched.begin() + pos, task_idx);
    }

    sol.worker_of_task[task_idx] = worker_id;
    recompute_worker_schedule(sol, worker_id);
}

bool RCPSPSolver::find_best_insertion(
    const Solution& sol,
    int task_idx,
    int& best_worker,
    int& best_pos,
    int& best_start,
    int& best_setup,
    double& best_delta_cost
) const {
    best_worker = -1;
    best_pos = -1;
    best_start = 0;
    best_setup = 0;
    best_delta_cost = 1e9;

    const auto& task = dag.tasks[task_idx];

    // Earliest start dictated by predecessors
    int pred_earliest = 0;
    for (int pred : task.predecessors) {
        if (sol.worker_of_task[pred] < 0) {
            return false; // Predecessor not yet scheduled
        }
        pred_earliest = std::max(pred_earliest, sol.end_of_task[pred]);
    }

    int num_workers = std::max(1, limits.max_concurrency);

    // Evaluate appending to each worker queue
    for (int w = 0; w < num_workers; ++w) {
        const auto& sched = sol.worker_schedule[w];
        int prev_task = sched.empty() ? -1 : sched.back();
        int setup = calculate_setup_cost(prev_task, task_idx);
        int worker_avail = sched.empty() ? 0 : sol.end_of_task[prev_task];
        int start = std::max(pred_earliest, worker_avail + setup);
        int end = start + task.duration_ms;

        // Affinity bonus: strongly prefer matching affinity key
        double affinity_discount = 0.0;
        if (prev_task >= 0 && !task.affinity_key.empty() && task.affinity_key == dag.tasks[prev_task].affinity_key) {
            affinity_discount = 300.0; // Bonus for session continuity
        }

        double delta = (end * weight_makespan) + (setup * weight_setup) - affinity_discount;
        if (delta < best_delta_cost) {
            best_delta_cost = delta;
            best_worker = w;
            best_pos = static_cast<int>(sched.size());
            best_start = start;
            best_setup = setup;
        }
    }

    return best_worker >= 0;
}

RCPSPSolver::Solution RCPSPSolver::construct_initial_greedy_solution() {
    Solution sol;
    size_t n = dag.tasks.size();
    sol.worker_of_task.assign(n, -1);
    sol.start_of_task.assign(n, 0);
    sol.end_of_task.assign(n, 0);
    sol.setup_of_task.assign(n, 0);
    sol.worker_schedule.resize(limits.max_concurrency);

    std::vector<int> topo = dag.topological_sort();
    for (int task_idx : topo) {
        int w, pos, start, setup;
        double cost;
        if (find_best_insertion(sol, task_idx, w, pos, start, setup, cost)) {
            insert_task_into_worker(sol, task_idx, w, pos, start, setup);
        }
    }

    evaluate_solution(sol);
    return sol;
}

// Destroy Operator 1: Critical Path Ruin
std::vector<int> RCPSPSolver::destroy_critical_path(
    Solution& sol, const TaskDAG::CPMResult& cpm, double fraction
) {
    std::vector<int> unassigned;
    for (int task_idx : cpm.critical_tasks) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng_) < fraction) {
            remove_task_from_solution(sol, task_idx);
            unassigned.push_back(task_idx);
        }
    }
    return unassigned;
}

// Destroy Operator 2: Resource Peak Ruin
std::vector<int> RCPSPSolver::destroy_resource_peak(Solution& sol, double fraction) {
    std::vector<int> unassigned;
    if (sol.makespan <= 0) return unassigned;

    // Sample a time point near peak
    int mid_point = sol.makespan / 2;
    int window = std::max(500, sol.makespan / 4);

    for (size_t i = 0; i < dag.tasks.size(); ++i) {
        if (sol.worker_of_task[i] >= 0) {
            int start = sol.start_of_task[i];
            if (std::abs(start - mid_point) < window) {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                if (dist(rng_) < fraction) {
                    remove_task_from_solution(sol, static_cast<int>(i));
                    unassigned.push_back(static_cast<int>(i));
                }
            }
        }
    }
    return unassigned;
}

// Destroy Operator 3: Worst Setup Ruin
std::vector<int> RCPSPSolver::destroy_worst_setup(Solution& sol, double fraction) {
    std::vector<std::pair<int, int>> setup_pairs;
    for (size_t i = 0; i < dag.tasks.size(); ++i) {
        if (sol.worker_of_task[i] >= 0) {
            setup_pairs.push_back({sol.setup_of_task[i], static_cast<int>(i)});
        }
    }
    std::sort(setup_pairs.rbegin(), setup_pairs.rend());

    size_t count = static_cast<size_t>(setup_pairs.size() * fraction);
    std::vector<int> unassigned;
    for (size_t i = 0; i < count && i < setup_pairs.size(); ++i) {
        int t = setup_pairs[i].second;
        remove_task_from_solution(sol, t);
        unassigned.push_back(t);
    }
    return unassigned;
}

// Destroy Operator 4: Domain Cluster Ruin
std::vector<int> RCPSPSolver::destroy_domain_cluster(Solution& sol) {
    std::unordered_map<std::string, std::vector<int>> clusters;
    for (size_t i = 0; i < dag.tasks.size(); ++i) {
        if (!dag.tasks[i].affinity_key.empty() && sol.worker_of_task[i] >= 0) {
            clusters[dag.tasks[i].affinity_key].push_back(static_cast<int>(i));
        }
    }

    std::vector<int> unassigned;
    if (clusters.empty()) return unassigned;

    // Pick the largest cluster
    auto max_it = std::max_element(clusters.begin(), clusters.end(),
        [](const auto& a, const auto& b) { return a.second.size() < b.second.size(); });

    for (int t : max_it->second) {
        remove_task_from_solution(sol, t);
        unassigned.push_back(t);
    }
    return unassigned;
}

// Repair Operator 1: Context Affinity Insertion
void RCPSPSolver::repair_context_affinity(Solution& sol, std::vector<int>& unassigned) {
    // Sort unassigned tasks by affinity key
    std::sort(unassigned.begin(), unassigned.end(), [this](int a, int b) {
        return dag.tasks[a].affinity_key < dag.tasks[b].affinity_key;
    });

    for (auto it = unassigned.begin(); it != unassigned.end();) {
        int t = *it;
        int w, pos, start, setup;
        double cost;
        if (find_best_insertion(sol, t, w, pos, start, setup, cost)) {
            insert_task_into_worker(sol, t, w, pos, start, setup);
            it = unassigned.erase(it);
        } else {
            ++it;
        }
    }
}

// Repair Operator 2: Regret-k Heuristic Insertion
void RCPSPSolver::repair_regret_k(Solution& sol, std::vector<int>& unassigned, int k) {
    while (!unassigned.empty()) {
        int best_task = -1;
        double max_regret = -1e9;
        int chosen_w = -1, chosen_pos = -1, chosen_start = 0, chosen_setup = 0;

        for (int t : unassigned) {
            // Find top-k insertion options
            std::vector<std::tuple<double, int, int, int, int>> options;
            int num_workers = std::max(1, limits.max_concurrency);

            int pred_earliest = 0;
            bool preds_ready = true;
            for (int pred : dag.tasks[t].predecessors) {
                if (sol.worker_of_task[pred] < 0) {
                    preds_ready = false;
                    break;
                }
                pred_earliest = std::max(pred_earliest, sol.end_of_task[pred]);
            }
            if (!preds_ready) continue;

            for (int w = 0; w < num_workers; ++w) {
                const auto& sched = sol.worker_schedule[w];
                int prev = sched.empty() ? -1 : sched.back();
                int setup = calculate_setup_cost(prev, t);
                int worker_avail = sched.empty() ? 0 : sol.end_of_task[prev];
                int start = std::max(pred_earliest, worker_avail + setup);
                int end = start + dag.tasks[t].duration_ms;
                double cost = (end * weight_makespan) + (setup * weight_setup);
                options.push_back({cost, w, static_cast<int>(sched.size()), start, setup});
            }

            if (options.empty()) continue;
            std::sort(options.begin(), options.end());

            double regret = 0.0;
            if (options.size() >= 2) {
                regret = std::get<0>(options[1]) - std::get<0>(options[0]);
            }

            if (regret > max_regret || best_task == -1) {
                max_regret = regret;
                best_task = t;
                chosen_w = std::get<1>(options[0]);
                chosen_pos = std::get<2>(options[0]);
                chosen_start = std::get<3>(options[0]);
                chosen_setup = std::get<4>(options[0]);
            }
        }

        if (best_task == -1) break;

        insert_task_into_worker(sol, best_task, chosen_w, chosen_pos, chosen_start, chosen_setup);
        unassigned.erase(std::remove(unassigned.begin(), unassigned.end(), best_task), unassigned.end());
    }
}

// Repair Operator 3: Earliest Feasible Fit
void RCPSPSolver::repair_earliest_feasible(Solution& sol, std::vector<int>& unassigned) {
    repair_context_affinity(sol, unassigned);
}

// Repair Operator 4: Priority Slack Insertion
void RCPSPSolver::repair_priority_slack(
    Solution& sol, std::vector<int>& unassigned, const TaskDAG::CPMResult& cpm
) {
    // Sort unassigned by slack ascending (critical tasks first)
    std::sort(unassigned.begin(), unassigned.end(), [&cpm](int a, int b) {
        return cpm.slack[a] < cpm.slack[b];
    });

    repair_context_affinity(sol, unassigned);
}

ScheduleResult RCPSPSolver::solve(int time_budget_ms) {
    auto start_time = std::chrono::steady_clock::now();

    auto cpm = dag.compute_critical_path();
    Solution best_sol = construct_initial_greedy_solution();
    Solution curr_sol = best_sol;

    // ALNS Operator weights & scores (4 Destroy, 4 Repair)
    std::vector<double> destroy_weights = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> repair_weights = {1.0, 1.0, 1.0, 1.0};

    int iterations = 0;
    while (true) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed >= time_budget_ms) {
            break;
        }

        iterations++;
        Solution candidate = curr_sol;

        // Roulette-wheel select destroy operator
        std::discrete_distribution<int> destroy_dist(destroy_weights.begin(), destroy_weights.end());
        int d_op = destroy_dist(rng_);
        std::vector<int> unassigned;

        switch (d_op) {
            case 0: unassigned = destroy_critical_path(candidate, cpm, 0.4); break;
            case 1: unassigned = destroy_resource_peak(candidate, 0.4); break;
            case 2: unassigned = destroy_worst_setup(candidate, 0.4); break;
            case 3: unassigned = destroy_domain_cluster(candidate); break;
        }

        // Roulette-wheel select repair operator
        std::discrete_distribution<int> repair_dist(repair_weights.begin(), repair_weights.end());
        int r_op = repair_dist(rng_);

        switch (r_op) {
            case 0: repair_context_affinity(candidate, unassigned); break;
            case 1: repair_regret_k(candidate, unassigned, 2); break;
            case 2: repair_earliest_feasible(candidate, unassigned); break;
            case 3: repair_priority_slack(candidate, unassigned, cpm); break;
        }

        // Fallback repair any residual unassigned tasks
        if (!unassigned.empty()) {
            repair_context_affinity(candidate, unassigned);
        }

        evaluate_solution(candidate);

        if (candidate.is_valid) {
            if (candidate.score < best_sol.score) {
                best_sol = candidate;
                curr_sol = candidate;
                destroy_weights[d_op] += 10.0;
                repair_weights[r_op] += 10.0;
            } else if (candidate.score < curr_sol.score) {
                curr_sol = candidate;
                destroy_weights[d_op] += 4.0;
                repair_weights[r_op] += 4.0;
            } else {
                // Decay
                destroy_weights[d_op] = std::max(0.2, destroy_weights[d_op] * 0.95);
                repair_weights[r_op] = std::max(0.2, repair_weights[r_op] * 0.95);
            }
        }
    }

    ScheduleResult result;
    result.makespan_ms = best_sol.makespan;
    result.total_setup_cost_ms = best_sol.total_setup_cost;
    result.objective_score = best_sol.score;
    result.feasible = best_sol.is_valid;
    result.iterations_completed = iterations;

    for (size_t i = 0; i < dag.tasks.size(); ++i) {
        ScheduleAssignment assign;
        assign.task_index = static_cast<int>(i);
        assign.task_id = dag.tasks[i].task_id;
        assign.worker_id = best_sol.worker_of_task[i];
        assign.start_time_ms = best_sol.start_of_task[i];
        assign.end_time_ms = best_sol.end_of_task[i];
        assign.setup_cost_ms = best_sol.setup_of_task[i];
        result.assignments.push_back(assign);
    }

    return result;
}

} // namespace agentic_or

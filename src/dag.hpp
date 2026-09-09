#pragma once

#include "types.hpp"
#include <vector>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <iostream>

namespace agentic_or {

class TaskDAG {
public:
    std::vector<TaskData> tasks;
    std::unordered_map<std::string, int> id_to_index;

    TaskDAG() = default;

    int add_task(const TaskData& task) {
        int idx = static_cast<int>(tasks.size());
        TaskData copy = task;
        copy.id = idx;
        tasks.push_back(copy);
        id_to_index[copy.task_id] = idx;
        return idx;
    }

    void add_dependency(int pred_idx, int succ_idx) {
        if (pred_idx < 0 || pred_idx >= static_cast<int>(tasks.size()) ||
            succ_idx < 0 || succ_idx >= static_cast<int>(tasks.size())) {
            throw std::out_of_range("Invalid task index for dependency");
        }
        tasks[pred_idx].successors.push_back(succ_idx);
        tasks[succ_idx].predecessors.push_back(pred_idx);
    }

    void add_dependency_by_id(const std::string& pred_id, const std::string& succ_id) {
        auto p_it = id_to_index.find(pred_id);
        auto s_it = id_to_index.find(succ_id);
        if (p_it == id_to_index.end() || s_it == id_to_index.end()) {
            throw std::invalid_argument("Task ID not found in DAG: " + pred_id + " -> " + succ_id);
        }
        add_dependency(p_it->second, s_it->second);
    }

    bool has_cycle() const {
        std::vector<int> in_degree(tasks.size(), 0);
        for (const auto& task : tasks) {
            in_degree[task.id] = static_cast<int>(task.predecessors.size());
        }

        std::queue<int> q;
        for (size_t i = 0; i < in_degree.size(); ++i) {
            if (in_degree[i] == 0) {
                q.push(static_cast<int>(i));
            }
        }

        int visited_count = 0;
        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            visited_count++;

            for (int succ : tasks[curr].successors) {
                in_degree[succ]--;
                if (in_degree[succ] == 0) {
                    q.push(succ);
                }
            }
        }

        return visited_count != static_cast<int>(tasks.size());
    }

    std::vector<int> topological_sort() const {
        if (has_cycle()) {
            throw std::runtime_error("DAG contains a cycle, cannot topological sort!");
        }

        std::vector<int> in_degree(tasks.size(), 0);
        for (const auto& task : tasks) {
            in_degree[task.id] = static_cast<int>(task.predecessors.size());
        }

        std::queue<int> q;
        for (size_t i = 0; i < in_degree.size(); ++i) {
            if (in_degree[i] == 0) {
                q.push(static_cast<int>(i));
            }
        }

        std::vector<int> order;
        order.reserve(tasks.size());

        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            order.push_back(curr);

            for (int succ : tasks[curr].successors) {
                in_degree[succ]--;
                if (in_degree[succ] == 0) {
                    q.push(succ);
                }
            }
        }

        return order;
    }

    struct CPMResult {
        std::vector<int> est;       // Earliest Start Time
        std::vector<int> eft;       // Earliest Finish Time
        std::vector<int> lst;       // Latest Start Time
        std::vector<int> lft;       // Latest Finish Time
        std::vector<int> slack;     // LST - EST
        std::vector<int> critical_tasks;
        int theoretical_min_makespan{0};
    };

    CPMResult compute_critical_path() const {
        if (tasks.empty()) {
            return {};
        }

        std::vector<int> order = topological_sort();
        size_t n = tasks.size();

        CPMResult res;
        res.est.assign(n, 0);
        res.eft.assign(n, 0);
        res.lst.assign(n, 0);
        res.lft.assign(n, 0);
        res.slack.assign(n, 0);

        // Forward Pass: Compute EST and EFT
        int max_eft = 0;
        for (int u : order) {
            int earliest = 0;
            for (int pred : tasks[u].predecessors) {
                earliest = std::max(earliest, res.eft[pred]);
            }
            res.est[u] = earliest;
            res.eft[u] = earliest + tasks[u].duration_ms;
            max_eft = std::max(max_eft, res.eft[u]);
        }
        res.theoretical_min_makespan = max_eft;

        // Backward Pass: Compute LFT and LST
        for (size_t i = 0; i < n; ++i) {
            res.lft[i] = max_eft;
        }

        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            int u = *it;
            if (!tasks[u].successors.empty()) {
                int min_succ_lst = std::numeric_limits<int>::max();
                for (int succ : tasks[u].successors) {
                    min_succ_lst = std::min(min_succ_lst, res.lst[succ]);
                }
                res.lft[u] = min_succ_lst;
            }
            res.lst[u] = res.lft[u] - tasks[u].duration_ms;
            res.slack[u] = res.lst[u] - res.est[u];
            if (res.slack[u] == 0) {
                res.critical_tasks.push_back(u);
            }
        }

        std::sort(res.critical_tasks.begin(), res.critical_tasks.end());
        return res;
    }
};

} // namespace agentic_or


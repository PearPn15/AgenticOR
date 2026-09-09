#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

namespace agentic_or {

enum class WorkloadType {
    LOCAL,
    API,
    BROWSER
};

inline WorkloadType parse_workload_type(const std::string& type_str) {
    if (type_str == "BROWSER") return WorkloadType::BROWSER;
    if (type_str == "API") return WorkloadType::API;
    return WorkloadType::LOCAL;
}

inline std::string workload_type_to_string(WorkloadType type) {
    switch (type) {
        case WorkloadType::BROWSER: return "BROWSER";
        case WorkloadType::API: return "API";
        case WorkloadType::LOCAL: return "LOCAL";
    }
    return "LOCAL";
}

struct TaskData {
    int id{0};                           // Integer index 0..N-1
    std::string task_id;                 // String identifier (e.g. "task_crawl_1")
    std::string name;
    WorkloadType workload_type{WorkloadType::LOCAL};
    int duration_ms{1000};               // Estimated duration in ms
    int ram_mb{100};                     // RAM usage in MB
    double cpu_percent{10.0};            // CPU percent 0..100
    int token_cost{0};                   // API token quota cost
    std::string affinity_key;            // Domain or session key for context reuse
    int deadline_ms{-1};                 // Optional deadline, -1 if unbounded
    std::vector<int> predecessors;       // Indices of tasks that must complete first
    std::vector<int> successors;         // Indices of tasks dependent on this
};

struct ResourceLimits {
    int max_ram_mb{8192};                // Safe ceiling for RAM usage
    double max_cpu_percent{85.0};        // Safe ceiling for CPU load
    int max_concurrency{4};              // Max concurrent workers
    int browser_cold_start_ms{2000};     // Cold start setup cost for browser worker
    int browser_tab_reuse_ms{150};       // Warm start context switch for same browser
    int local_setup_ms{20};              // Local worker setup cost
};

struct ScheduleAssignment {
    int task_index{-1};
    std::string task_id;
    int worker_id{-1};
    int start_time_ms{0};
    int end_time_ms{0};
    int setup_cost_ms{0};
};

struct ScheduleResult {
    std::vector<ScheduleAssignment> assignments;
    int makespan_ms{0};
    int total_setup_cost_ms{0};
    double objective_score{0.0};
    bool feasible{false};
    int iterations_completed{0};
};

} // namespace agentic_or


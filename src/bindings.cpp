#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "types.hpp"
#include "dag.hpp"
#include "alns.hpp"
#include "bandit.cpp"

namespace py = pybind11;
using namespace agentic_or;

PYBIND11_MODULE(_cxx_engine, m) {
    m.doc() = "High-Performance C++20 Optimization Solver Engine for Desktop-Agent-OR";

    // WorkloadType enum
    py::enum_<WorkloadType>(m, "WorkloadType")
        .value("LOCAL", WorkloadType::LOCAL)
        .value("API", WorkloadType::API)
        .value("BROWSER", WorkloadType::BROWSER)
        .export_values();

    // TaskData struct
    py::class_<TaskData>(m, "TaskData")
        .def(py::init<>())
        .def_readwrite("id", &TaskData::id)
        .def_readwrite("task_id", &TaskData::task_id)
        .def_readwrite("name", &TaskData::name)
        .def_readwrite("workload_type", &TaskData::workload_type)
        .def_readwrite("duration_ms", &TaskData::duration_ms)
        .def_readwrite("ram_mb", &TaskData::ram_mb)
        .def_readwrite("cpu_percent", &TaskData::cpu_percent)
        .def_readwrite("token_cost", &TaskData::token_cost)
        .def_readwrite("affinity_key", &TaskData::affinity_key)
        .def_readwrite("deadline_ms", &TaskData::deadline_ms)
        .def_readwrite("predecessors", &TaskData::predecessors)
        .def_readwrite("successors", &TaskData::successors);

    // ResourceLimits struct
    py::class_<ResourceLimits>(m, "ResourceLimits")
        .def(py::init<>())
        .def_readwrite("max_ram_mb", &ResourceLimits::max_ram_mb)
        .def_readwrite("max_cpu_percent", &ResourceLimits::max_cpu_percent)
        .def_readwrite("max_concurrency", &ResourceLimits::max_concurrency)
        .def_readwrite("browser_cold_start_ms", &ResourceLimits::browser_cold_start_ms)
        .def_readwrite("browser_tab_reuse_ms", &ResourceLimits::browser_tab_reuse_ms)
        .def_readwrite("local_setup_ms", &ResourceLimits::local_setup_ms);

    // ScheduleAssignment struct
    py::class_<ScheduleAssignment>(m, "ScheduleAssignment")
        .def(py::init<>())
        .def_readwrite("task_index", &ScheduleAssignment::task_index)
        .def_readwrite("task_id", &ScheduleAssignment::task_id)
        .def_readwrite("worker_id", &ScheduleAssignment::worker_id)
        .def_readwrite("start_time_ms", &ScheduleAssignment::start_time_ms)
        .def_readwrite("end_time_ms", &ScheduleAssignment::end_time_ms)
        .def_readwrite("setup_cost_ms", &ScheduleAssignment::setup_cost_ms);

    // ScheduleResult struct
    py::class_<ScheduleResult>(m, "ScheduleResult")
        .def(py::init<>())
        .def_readwrite("assignments", &ScheduleResult::assignments)
        .def_readwrite("makespan_ms", &ScheduleResult::makespan_ms)
        .def_readwrite("total_setup_cost_ms", &ScheduleResult::total_setup_cost_ms)
        .def_readwrite("objective_score", &ScheduleResult::objective_score)
        .def_readwrite("feasible", &ScheduleResult::feasible)
        .def_readwrite("iterations_completed", &ScheduleResult::iterations_completed);

    // TaskDAG::CPMResult struct
    py::class_<TaskDAG::CPMResult>(m, "CPMResult")
        .def(py::init<>())
        .def_readwrite("est", &TaskDAG::CPMResult::est)
        .def_readwrite("eft", &TaskDAG::CPMResult::eft)
        .def_readwrite("lst", &TaskDAG::CPMResult::lst)
        .def_readwrite("lft", &TaskDAG::CPMResult::lft)
        .def_readwrite("slack", &TaskDAG::CPMResult::slack)
        .def_readwrite("critical_tasks", &TaskDAG::CPMResult::critical_tasks)
        .def_readwrite("theoretical_min_makespan", &TaskDAG::CPMResult::theoretical_min_makespan);

    // TaskDAG class
    py::class_<TaskDAG>(m, "TaskDAG")
        .def(py::init<>())
        .def("add_task", &TaskDAG::add_task)
        .def("add_dependency", &TaskDAG::add_dependency)
        .def("add_dependency_by_id", &TaskDAG::add_dependency_by_id)
        .def("has_cycle", &TaskDAG::has_cycle)
        .def("topological_sort", &TaskDAG::topological_sort)
        .def("compute_critical_path", &TaskDAG::compute_critical_path)
        .def_readwrite("tasks", &TaskDAG::tasks);

    // RCPSPSolver class
    py::class_<RCPSPSolver>(m, "RCPSPSolver")
        .def(py::init<const TaskDAG&, const ResourceLimits&>())
        .def("set_weights", &RCPSPSolver::set_weights)
        .def("solve", &RCPSPSolver::solve, py::arg("time_budget_ms") = 150);

    // LinearContextualBandit class
    py::class_<LinearContextualBandit>(m, "LinearContextualBandit")
        .def(py::init<double>(), py::arg("exploration_alpha") = 0.2)
        .def("reset", &LinearContextualBandit::reset)
        .def("select_action", &LinearContextualBandit::select_action)
        .def("update", &LinearContextualBandit::update)
        .def("action_to_concurrency", &LinearContextualBandit::action_to_concurrency);
}


#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <array>

namespace agentic_or {

class LinearContextualBandit {
public:
    static constexpr int D = 6;     // Feature dimension
    static constexpr int K = 4;     // Actions: 0 -> cap 2, 1 -> cap 4, 2 -> cap 6, 3 -> cap 8

    double alpha{0.2};              // Exploration parameter for LinUCB

    // Per-arm matrices: A_a (D x D), b_a (D x 1)
    std::array<std::array<std::array<double, D>, D>, K> A_;
    std::array<std::array<double, D>, K> b_;

    LinearContextualBandit(double exploration_alpha = 0.2)
        : alpha(exploration_alpha) {
        reset();
    }

    void reset() {
        for (int a = 0; a < K; ++a) {
            for (int i = 0; i < D; ++i) {
                b_[a][i] = 0.0;
                for (int j = 0; j < D; ++j) {
                    A_[a][i][j] = (i == j) ? 1.0 : 0.0; // Identity matrix
                }
            }
        }
    }

    int select_action(const std::vector<double>& context) const {
        if (static_cast<int>(context.size()) != D) {
            throw std::invalid_argument("Context vector dimension must be 6");
        }

        int best_arm = 0;
        double max_ucb = -1e9;

        for (int a = 0; a < K; ++a) {
            // Invert A_a (6x6 using Gauss-Jordan)
            auto inv_A = invert_matrix(A_[a]);

            // theta_a = inv_A * b_a
            std::array<double, D> theta{};
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    theta[i] += inv_A[i][j] * b_[a][j];
                }
            }

            // Expected value = theta^T * x
            double mean = 0.0;
            for (int i = 0; i < D; ++i) {
                mean += theta[i] * context[i];
            }

            // Variance term = sqrt(x^T * inv_A * x)
            double var = 0.0;
            for (int i = 0; i < D; ++i) {
                double temp = 0.0;
                for (int j = 0; j < D; ++j) {
                    temp += inv_A[i][j] * context[j];
                }
                var += context[i] * temp;
            }
            double bonus = alpha * std::sqrt(std::max(0.0, var));
            double ucb = mean + bonus;

            if (ucb > max_ucb) {
                max_ucb = ucb;
                best_arm = a;
            }
        }

        return best_arm;
    }

    void update(const std::vector<double>& context, int action, double reward) {
        if (static_cast<int>(context.size()) != D || action < 0 || action >= K) {
            return;
        }

        // A_a += x * x^T
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                A_[action][i][j] += context[i] * context[j];
            }
            // b_a += r * x
            b_[action][i] += reward * context[i];
        }
    }

    int action_to_concurrency(int action) const {
        switch (action) {
            case 0: return 2;
            case 1: return 4;
            case 2: return 6;
            case 3: return 8;
            default: return 4;
        }
    }

private:
    // Simple 6x6 Gauss-Jordan elimination matrix inverter
    std::array<std::array<double, D>, D> invert_matrix(
        const std::array<std::array<double, D>, D>& m
    ) const {
        std::array<std::array<double, D>, D> inv{};
        std::array<std::array<double, 2 * D>, D> aug{};

        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                aug[i][j] = m[i][j];
                aug[i][j + D] = (i == j) ? 1.0 : 0.0;
            }
        }

        for (int i = 0; i < D; ++i) {
            // Pivot selection
            int pivot = i;
            for (int r = i + 1; r < D; ++r) {
                if (std::abs(aug[r][i]) > std::abs(aug[pivot][i])) {
                    pivot = r;
                }
            }
            if (pivot != i) {
                std::swap(aug[i], aug[pivot]);
            }

            double diag = aug[i][i];
            if (std::abs(diag) < 1e-9) diag = 1e-9;

            for (int c = 0; c < 2 * D; ++c) {
                aug[i][c] /= diag;
            }

            for (int r = 0; r < D; ++r) {
                if (r != i) {
                    double factor = aug[r][i];
                    for (int c = 0; c < 2 * D; ++c) {
                        aug[r][c] -= factor * aug[i][c];
                    }
                }
            }
        }

        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                inv[i][j] = aug[i][j + D];
            }
        }
        return inv;
    }
};

} // namespace agentic_or


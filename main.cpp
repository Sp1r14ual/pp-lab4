#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>

constexpr double a = 10e5;
constexpr double epsilon = 10e-6;
constexpr int max_iterations = 20;
constexpr double initial_guess = 0.0;

const std::vector<double> domain_start{-1.0, -1.0, -1.0};
const std::vector<double> domain_size{2.0, 2.0, 2.0};
std::vector<int> grid_dimensions{480, 480, 480};

struct GridParameters
{
    std::vector<double> steps;
    int local_z_size;

    GridParameters(int world_size)
    {
        steps = {
            domain_size[0] / (grid_dimensions[0] - 1),
            domain_size[1] / (grid_dimensions[1] - 1),
            domain_size[2] / (grid_dimensions[2] - 1)};
        local_z_size = grid_dimensions[2] / world_size;
        grid_dimensions[2] = local_z_size;
    }
};

double exact_solution(double x, double y, double z)
{
    return x * x + y * y + z * z;
}

double source_term(double x, double y, double z)
{
    return 6.0 - a * exact_solution(x, y, z);
}

double compute_next_value(const std::vector<std::vector<double>> &prev,
                          int k, int i, const GridParameters &params, int rank)
{
    const double hx_sq = params.steps[0] * params.steps[0];
    const double hy_sq = params.steps[1] * params.steps[1];
    const double hz_sq = params.steps[2] * params.steps[2];

    const double denominator = 2.0 / hx_sq + 2.0 / hy_sq + 2.0 / hz_sq + a;

    const int x_dim = grid_dimensions[0];
    const int y_dim = grid_dimensions[1];

    const double x = domain_start[0] + (i % x_dim) * params.steps[0];
    const double y = domain_start[1] + (i / x_dim % y_dim) * params.steps[1];
    const double z = domain_start[2] + (k + rank * params.local_z_size) * params.steps[2];

    return ((prev[k][i + 1] + prev[k][i - 1]) / hx_sq +
            (prev[k][i + x_dim] + prev[k][i - x_dim]) / hy_sq +
            (prev[k + 1][i] + prev[k - 1][i]) / hz_sq -
            source_term(x, y, z)) /
           denominator;
}

void exchange_ghost_layers(std::vector<std::vector<double>> &phi,
                           int rank, int size, MPI_Request *requests)
{
    const int NxNy = grid_dimensions[0] * grid_dimensions[1];

    if (rank != 0)
    {
        MPI_Isend(phi[1].data(), NxNy, MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(phi[0].data(), NxNy, MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &requests[1]);
    }

    if (rank != size - 1)
    {
        MPI_Isend(phi[phi.size() - 2].data(), NxNy, MPI_DOUBLE, rank + 1, 0,
                  MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(phi.back().data(), NxNy, MPI_DOUBLE, rank + 1, 0,
                  MPI_COMM_WORLD, &requests[3]);
    }
}

void initialize_grid(std::vector<std::vector<double>> &phi, int rank,
                     const GridParameters &params)
{
    const int x_dim = grid_dimensions[0];
    const int y_dim = grid_dimensions[1];

    for (int k = 0; k < phi.size(); ++k)
    {
        for (int i = 0; i < phi[k].size(); ++i)
        {
            const int x_idx = i % x_dim;
            const int y_idx = (i / x_dim) % y_dim;
            const double z = domain_start[2] + (k + rank * params.local_z_size) * params.steps[2];

            if (x_idx == 0 || x_idx == x_dim - 1 ||
                y_idx == 0 || y_idx == y_dim - 1 ||
                k == 0 || k == phi.size() - 1)
            {
                phi[k][i] = exact_solution(
                    domain_start[0] + x_idx * params.steps[0],
                    domain_start[1] + y_idx * params.steps[1],
                    z);
            }
            else
            {
                phi[k][i] = initial_guess;
            }
        }
    }
}

double calculate_max_error(const std::vector<std::vector<double>> &curr,
                           const std::vector<std::vector<double>> &prev)
{
    double local_max = 0.0;
    for (size_t k = 0; k < curr.size(); ++k)
    {
        for (size_t i = 0; i < curr[k].size(); ++i)
        {
            local_max = std::max(local_max, std::abs(curr[k][i] - prev[k][i]));
        }
    }

    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    GridParameters params(world_size);
    std::vector<std::vector<double>> phi(params.local_z_size,
                                         std::vector<double>(grid_dimensions[0] * grid_dimensions[1]));
    auto phi_prev = phi;

    initialize_grid(phi, world_rank, params);

    double error = 1.0;
    int iteration = 1;
    MPI_Request requests[4];

    const double start_time = MPI_Wtime();

    while (error > epsilon && iteration <= max_iterations)
    {
        phi_prev = phi;

        // Boundary layer updates
        for (int k = 1; k < phi.size() - 1; ++k)
        {
            for (int i = grid_dimensions[0] + 1; i < phi[k].size() - grid_dimensions[0] - 1; ++i)
            {
                phi[k][i] = compute_next_value(phi_prev, k, i, params, world_rank);
            }
        }

        exchange_ghost_layers(phi, world_rank, world_size, requests);

        // Internal points update
        for (int k = 2; k < phi.size() - 2; ++k)
        {
            for (int i = 2 * grid_dimensions[0] + 1; i < phi[k].size() - 2 * grid_dimensions[0] - 1; ++i)
            {
                phi[k][i] = compute_next_value(phi_prev, k, i, params, world_rank);
            }
        }

        MPI_Status statuses[4];
        if (world_rank != 0)
            MPI_Waitall(2, requests, statuses);
        if (world_rank != world_size - 1)
            MPI_Waitall(2, requests + 2, statuses + 2);

        error = calculate_max_error(phi, phi_prev);
        if (world_rank == 0)
        {
            std::cout << "Iteration " << iteration
                      << ", Error: " << error << std::endl;
        }
        iteration++;
    }

    const double end_time = MPI_Wtime();

    if (world_rank == 0)
    {
        std::cout << "Total time: " << end_time - start_time << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
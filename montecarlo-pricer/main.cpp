#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

double normalCDF(double x)
{
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double blackScholesCall(double S0, double K, double r, double sigma, double T)
{
    double d1 = (std::log(S0 / K) + (r + (sigma * sigma) / 2) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    double C = S0 * normalCDF(d1) - K * std::exp(-r * T) * normalCDF(d2);
    return C;
}

double monteCarloCall(double S0, double K, double r, double sigma, double T, int numPaths)
{
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    double sum = 0;

    for (int i = 0; i < numPaths; i++)
    {
        double Z = normal(rng);
        double S_T = S0 * std::exp((r - sigma * sigma / 2) * T + sigma * std::sqrt(T) * Z);
        double payoff = std::max(S_T - K, 0.0);
        sum += payoff;
    }

    return std::exp(-r * T) * (sum / numPaths);
}

int main()
{
    std::cout << "Black-Scholes: " << blackScholesCall(100, 100, 0.05, 0.2, 2) << "\n";
    std::cout << "Monte Carlo:   " << monteCarloCall(100, 100, 0.05, 0.2, 2, 1000000) << "\n";
    return 0;


}
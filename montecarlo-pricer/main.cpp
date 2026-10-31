#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

struct MCResult
{
    double price;
    double standardError;
};

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

MCResult monteCarloCall(double S0, double K, double r, double sigma, double T, int numPaths)
{
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    double sum = 0;
    double sumSquared = 0;
    double discount = std::exp(-r * T);

    for (int i = 0; i < numPaths; i++)
    {
        double Z = normal(rng);
        double S_T = S0 * std::exp((r - sigma * sigma / 2) * T + sigma * std::sqrt(T) * Z);
        double payoff = std::max(S_T - K, 0.0);
        double discounted = discount * payoff;
        sum += discounted;
        sumSquared += discounted * discounted;
        
    }

    double variance = (sumSquared / numPaths) - (sum / numPaths) * (sum / numPaths);
    double mean = sum / numPaths;
    double s = std::sqrt(variance)/ std::sqrt(numPaths);
    return MCResult{ mean, s };
}

int main()
{
    double exact = blackScholesCall(100, 100, 0.05, 0.2, 1);
    std::cout << std::setw(12) << "N" << std::setw(12) << "Price" << std::setw(12) << "Std Err" << std::setw(12) << "Actual Err" << "\n";
    for (int N = 1000; N <= 1000000; N *= 10)
    {
        MCResult result = monteCarloCall(100, 100, 0.05, 0.2, 1, N);
        std::cout << std::setw(12) << N
            << std::setw(12) << result.price
            << std::setw(12) << result.standardError
            << std::setw(12) << std::abs(result.price - exact) << "\n";
    }
  
    return 0;
}
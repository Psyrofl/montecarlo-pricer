#include <iostream>
#include <cmath>

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

int main()
{
    std::cout << blackScholesCall(100, 100, 0.05, 0.2, 1) << "\n";   
    std::cout << blackScholesCall(100, 100, 0.05, 0.2, 2) << "\n";   
    return 0;
}
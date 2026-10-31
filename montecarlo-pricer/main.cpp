#include <iostream>
#include <cmath>

double normalCDF(double x)
{
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double blackScholesCall(double S0, double K, double r, double sigma, double T)
{
    // your code: d1, d2, then return C
}

int main()
{
    std::cout << blackScholesCall(100, 100, 0.05, 0.2, 1) << "\n";
    return 0;
}
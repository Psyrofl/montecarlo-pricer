#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>

struct ThreadTotals
{
    double sum = 0;
    double sumSquared = 0;
};

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

    double mean = sum / numPaths;
    double variance = (sumSquared / numPaths) - mean * mean;
    double s = std::sqrt(variance)/ std::sqrt(numPaths);
    return MCResult{ mean, s };
}

void simulatePaths(double S0, double K, double r, double sigma, double T, int numPaths, unsigned seed, ThreadTotals& out)
{
    std::mt19937 rng(seed);    
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

    out.sum = sum;
    out.sumSquared = sumSquared;
   
}

MCResult monteCarloCallParallel(double S0, double K, double r, double sigma, double T, int numPaths, int numThreads)
{
    std::vector<ThreadTotals> totals(numThreads);
    std::vector<std::thread> threads;
    int pathsPerThread = numPaths / numThreads;
    double sum = 0;
    double sumSquared = 0;

    for (int i = 0; i < numThreads; i++)
    {
        int pathsForThisThread = pathsPerThread;

        if (i == numThreads - 1)
        {
            pathsForThisThread = numPaths - pathsPerThread * (numThreads - 1);
        }
       
        threads.push_back(std::thread(simulatePaths, S0, K, r, sigma, T, pathsForThisThread, 42 + i, std::ref(totals[i])));
    }

    for (std::thread& t : threads)
    {
        t.join();
    }

    for (int i = 0; i < totals.size(); i++)
    {
        sum += totals[i].sum;
        sumSquared += totals[i].sumSquared;
    }

    double mean = sum / numPaths;
    double variance = (sumSquared / numPaths) - mean * mean;

    double s = std::sqrt(variance) / std::sqrt(numPaths);
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

    const int paths = 10000000;
    double baseTime = 0;

    std::cout << "\n" << std::setw(12) << "Threads" << std::setw(12) << "Time (s)" << std::setw(12) << "Speedup" << std::setw(12) << "Price" << std::setw(12) << "Std Err" << "\n";

    for (int threads : {1, 2, 4, 8, 16})
    {
        auto start = std::chrono::steady_clock::now();
        MCResult result = monteCarloCallParallel(100, 100, 0.05, 0.2, 1, paths, threads);
        auto end = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();

        if (threads == 1)
        {
            baseTime = seconds;
        }

        std::cout << std::setw(12) << threads
            << std::setw(12) << seconds
            << std::setw(12) << baseTime / seconds
            << std::setw(12) << result.price
            << std::setw(12) << result.standardError << "\n";
    }
  
    return 0;
}
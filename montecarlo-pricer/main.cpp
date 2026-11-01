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

double normalCDF(double x);
double estimateBeta(double S0, double K, double r, double sigma, double T, int numPilot, unsigned seed);
double blackScholesCall(double S0, double K, double r, double sigma, double T);
MCResult monteCarloCall(double S0, double K, double r, double sigma, double T, int numPaths);
void simulatePaths(double S0, double K, double r, double sigma, double T, int numPaths, unsigned seed, ThreadTotals& out);
void simulatePathsAntithetic(double S0, double K, double r, double sigma, double T, int numPaths, unsigned seed, ThreadTotals& out);
void simulatePathsControl(double S0, double K, double r, double sigma, double T, int numPaths, unsigned seed, double beta, ThreadTotals& out);
void simulatePathsAsian(double S0, double K, double r, double sigma, double T, int numPaths, int numSteps, unsigned seed, ThreadTotals& out);
MCResult monteCarloCallParallel(double S0, double K, double r, double sigma, double T, int numPaths, int numThreads);
MCResult monteCarloCallAntithetic(double S0, double K, double r, double sigma, double T, int numPaths, int numThreads);
MCResult monteCarloCallControl(double S0, double K, double r, double sigma, double T, int numPaths, int numThreads);
MCResult monteCarloCallAsian(double S0, double K, double r, double sigma, double T, int numPaths, int numSteps, int numThreads);


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

    MCResult standard = monteCarloCallParallel(100, 100, 0.05, 0.2, 1, paths, 16);
    MCResult antithetic = monteCarloCallAntithetic(100, 100, 0.05, 0.2, 1, paths, 16);
    MCResult control = monteCarloCallControl(100, 100, 0.05, 0.2, 1, paths, 16);
    MCResult asian1 = monteCarloCallAsian(100, 100, 0.05, 0.2, 1, 1000000, 1, 16);
    MCResult asian252 = monteCarloCallAsian(100, 100, 0.05, 0.2, 1, 1000000, 252, 16);

    double antitheticRatio = standard.standardError / antithetic.standardError;
    double controlRatio = standard.standardError / control.standardError;

    std::cout << "\nExact:      " << exact << "\n";
    std::cout << "Standard:   " << standard.price << " +/- " << standard.standardError << "\n";
    std::cout << "Antithetic: " << antithetic.price << " +/- " << antithetic.standardError << "  factor " << antitheticRatio * antitheticRatio << "\n";
    std::cout << "Control:    " << control.price << " +/- " << control.standardError << "  factor " << controlRatio * controlRatio << "\n";
    std::cout << "Asian, 1 step:    " << asian1.price << " +/- " << 1.96 * asian1.standardError << "\n";
    std::cout << "Asian, 252 steps: " << asian252.price << " +/- " << 1.96 * asian252.standardError << "\n";
    return 0;
}

double normalCDF(double x)
{
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double estimateBeta(double S0, double K, double r, double sigma, double T, int numPilot, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double sumX = 0;
    double sumY = 0;
    double sumXX = 0;
    double sumXY = 0;



    double discount = std::exp(-r * T);

    for (int i = 0; i < numPilot; i++)
    {
        double Z = normal(rng);
        double S_T = S0 * std::exp((r - sigma * sigma / 2) * T + sigma * std::sqrt(T) * Z);
        double payoff = std::max(S_T - K, 0.0);
        double stockavg = std::exp(-r * T) * S_T;
        double discounted = discount * payoff;
        sumX += stockavg;
        sumY += discounted;
        sumXY += stockavg * discounted;
        sumXX += stockavg * stockavg;
    }

    double covXY = sumXY / numPilot - (sumX / numPilot) * (sumY / numPilot);
    double varX = sumXX / numPilot - (sumX / numPilot) * (sumX / numPilot);
    double beta = covXY / varX;
    return beta;
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
    double s = std::sqrt(variance) / std::sqrt(numPaths);
    return MCResult{ mean, s };
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

MCResult monteCarloCallAntithetic(double S0, double K, double r, double sigma, double T, int numPaths, int numThreads)
{
    std::vector<ThreadTotals> totals(numThreads);
    std::vector<std::thread> threads;
    int numPairs = numPaths / 2;
    int pathsPerThread = numPairs / numThreads;
    double sum = 0;
    double sumSquared = 0;

    for (int i = 0; i < numThreads; i++)
    {
        int pathsForThisThread = pathsPerThread;

        if (i == numThreads - 1)
        {
            pathsForThisThread = numPairs - pathsPerThread * (numThreads - 1);
        }

        threads.push_back(std::thread(simulatePathsAntithetic, S0, K, r, sigma, T, pathsForThisThread, 42 + i, std::ref(totals[i])));
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

    double mean = sum / numPairs;
    double variance = (sumSquared / numPairs) - mean * mean;

    double s = std::sqrt(variance) / std::sqrt(numPairs);
    return MCResult{ mean, s };
}

MCResult monteCarloCallControl(double S0, double K, double r, double sigma, double T, int numPaths, int numThreads)
{
    double beta = estimateBeta(S0, K, r, sigma, T, 100000, 1000);
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

        threads.push_back(std::thread(simulatePathsControl, S0, K, r, sigma, T, pathsForThisThread, 42 + i, beta, std::ref(totals[i])));
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

MCResult monteCarloCallAsian(double S0, double K, double r, double sigma, double T, int numPaths, int numSteps, int numThreads)
{
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

            threads.push_back(std::thread(simulatePathsAsian, S0, K, r, sigma, T, pathsForThisThread, numSteps, 42 + i, std::ref(totals[i])));
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

void simulatePathsAntithetic(double S0, double K, double r, double sigma, double T, int numPaths, unsigned seed, ThreadTotals& out)
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
        double S_T_2 = S0 * std::exp((r - sigma * sigma / 2) * T + sigma * std::sqrt(T) * -Z);
        double payoff2 = std::max(S_T_2 - K, 0.0);
        double payoffavg = (payoff + payoff2) / 2;
        double discounted = discount * payoffavg;
        sum += discounted;
        sumSquared += discounted * discounted;

    }

    out.sum = sum;
    out.sumSquared = sumSquared;

}

void simulatePathsControl(double S0, double K, double r, double sigma, double T, int numPaths, unsigned seed, double beta, ThreadTotals& out)
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
        double X = discount * S_T;
        double corrected = discounted - beta * (X - S0);
        sum += corrected;
        sumSquared += corrected * corrected;

    }

    out.sum = sum;
    out.sumSquared = sumSquared;

}

void simulatePathsAsian(double S0, double K, double r, double sigma, double T, int numPaths, int numSteps, unsigned seed, ThreadTotals& out)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double discount = std::exp(-r * T);
    double dt = T / numSteps;

    for (int i = 0; i < numPaths; i++)
    {
        double S = S0;
        double priceSum = 0;

        for (int step = 0; step < numSteps; step++)
        {
            double Z = normal(rng);
            S = S * exp((r - sigma * sigma / 2) * dt + sigma * std::sqrt(dt) * Z);
            priceSum += S;
            
        }
        double average = priceSum / numSteps;
        double payoff = std::max(average - K, 0.0);
        double discounted = discount * payoff;
        out.sum += discounted;
        out.sumSquared += discounted * discounted;
    }
}






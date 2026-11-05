#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include "payoff.h"
#include "sampling.h"

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
void simulatePathsGeneric(const Payoff& payoff, double S0, double r, double sigma, double T, int numPaths, int numSteps, unsigned seed, ThreadTotals& out);
void simulatePathsHalton(const Payoff& payoff, double S0, double r, double sigma, double T, int numPaths, int numSteps, int startIndex, const std::vector<int>& bases, ThreadTotals& out);
MCResult monteCarloCallParallel(double S0, double K, double r, double sigma, double T, int numPaths, int numThreads);
MCResult monteCarloCallAntithetic(double S0, double K, double r, double sigma, double T, int numPaths, int numThreads);
MCResult monteCarloCallControl(double S0, double K, double r, double sigma, double T, int numPaths, int numThreads);
MCResult monteCarloCallHalton(double S0, double K, double r, double sigma, double T, int numPaths);
MCResult monteCarloPrice(const Payoff& payoff, double S0, double r, double sigma, double T, int numPaths, int numSteps, int numThreads);
MCResult monteCarloPriceHalton(const Payoff& payoff, double S0, double r, double sigma, double T, int numPaths, int numSteps, int numThreads);
double impliedVol(double marketPrice, double S0, double K, double r, double T);
void impliedVolTable(const std::string& filename, double S0, double r, double T);



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
   
     
    double antitheticRatio = standard.standardError / antithetic.standardError;
    double controlRatio = standard.standardError / control.standardError;

    std::cout << "\nExact:      " << exact << "\n";
    std::cout << "Standard:   " << standard.price << " +/- " << standard.standardError << "\n";
    std::cout << "Antithetic: " << antithetic.price << " +/- " << antithetic.standardError << "  factor " << antitheticRatio * antitheticRatio << "\n";
    std::cout << "Control:    " << control.price << " +/- " << control.standardError << "  factor " << controlRatio * controlRatio << "\n";


    AsianPayoff asian(100);
    MCResult asian1 = monteCarloPrice(asian, 100, 0.05, 0.2, 1, 1000000, 1, 16);
    MCResult asian252 = monteCarloPrice(asian, 100, 0.05, 0.2, 1, 1000000, 252, 16);

    std::cout << "Asian, 1 step:    " << asian1.price << " +/- " << asian1.standardError << "\n";
    std::cout << "Asian, 252 steps: " << asian252.price << " +/- " << asian252.standardError << "\n";

    for (double barrier : {1000000.0, 150.0, 120.0, 100.0})
    {
        BarrierPayoff payoff(100, barrier);
        MCResult result = monteCarloPrice(payoff, 100, 0.05, 0.2, 1, 1000000, 252, 16);
        std::cout << "Barrier " << barrier << ": " << result.price << " +/- " << result.standardError << "\n";
    }

    LookbackPayoff lookback(100);
    MCResult lookback1 = monteCarloPrice(lookback, 100, 0.05, 0.2, 1, 1000000, 1, 16);
    MCResult lookback252 = monteCarloPrice(lookback, 100, 0.05, 0.2, 1, 1000000, 252, 16);
    std::cout << "Lookback, 1 step:    " << lookback1.price << " +/- " << lookback1.standardError << "\n";
    std::cout << "Lookback, 252 steps: " << lookback252.price << " +/- " << lookback252.standardError << "\n";


    for (int i = 1; i <= 8; i++)
    {
        std::cout << halton(i, 2) << " ";
    }

    for (double Z : {-3.0, -1.0, 0.0, 0.5, 2.0})
    {
        std::cout << Z << " -> " << inverseNormalCDF(normalCDF(Z)) << "\n";
    }

    std::cout << "\n" << std::setw(12) << "N" << std::setw(16) << "Random error" << std::setw(16) << "Halton error" << "\n";
    for (int N = 1000; N <= 1000000; N *= 10)
    {
        MCResult random = monteCarloCall(100, 100, 0.05, 0.2, 1, N);
        MCResult halton = monteCarloCallHalton(100, 100, 0.05, 0.2, 1, N);
        std::cout << std::setw(12) << N
            << std::setw(16) << std::abs(random.price - exact)
            << std::setw(16) << std::abs(halton.price - exact) << "\n";
    }

    std::vector<int> p = firstPrimes(252);
    for (int i = 0; i < 10; i++) std::cout << p[i] << " ";
    std::cout << "... " << p.back() << "\n";

    std::cout << "\n" << std::setw(6) << "Steps" << std::setw(16) << "Random error" << std::setw(16) << "Halton error" << "\n";
    VanillaPayoff vanilla(100);
    for (int steps : {1, 12, 52, 252})
    {
        MCResult random = monteCarloPrice(vanilla, 100, 0.05, 0.2, 1, 100000, steps, 16);
        MCResult halton = monteCarloPriceHalton(vanilla, 100, 0.05, 0.2, 1, 100000, steps, 16);
        std::cout << std::setw(6) << steps
            << std::setw(16) << std::abs(random.price - exact)
            << std::setw(16) << std::abs(halton.price - exact) << "\n";
    }

    for (double sigma : {0.1, 0.2, 0.35, 0.6})
    {
        double price = blackScholesCall(100, 100, 0.05, sigma, 1);
        std::cout << sigma << " -> " << impliedVol(price, 100, 100, 0.05, 1) << "\n";
    }

    impliedVolTable("spx_calls.csv", 7688.62, 0.0413, 64.0 / 365);

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

MCResult monteCarloCallHalton(double S0, double K, double r, double sigma, double T, int numPaths)
{
    double sum = 0;
    double sumSquared = 0;
    double discount = std::exp(-r * T);

    for (int i = 0; i < numPaths; i++)
    {
        double Z = inverseNormalCDF(halton(i + 1, 2));
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

void simulatePathsGeneric(const Payoff& payoff, double S0, double r, double sigma, double T, int numPaths, int numSteps, unsigned seed, ThreadTotals& out)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double discount = std::exp(-r * T);
    double dt = T / numSteps;
    double sum = 0;
    double sumSquared = 0;
    std::vector<double> path(numSteps);
    for (int i = 0; i < numPaths; i++)
    {
        double S = S0;
        for (int step = 0; step < numSteps; step++)
        {
            double Z = normal(rng);
            S = S * std::exp((r - sigma * sigma / 2) * dt + sigma * std::sqrt(dt) * Z);
            path[step] = S;

        }
        double discounted = payoff.calculate(path) * discount;          
        sum += discounted;
        sumSquared += discounted * discounted;
    }

    out.sum = sum;
    out.sumSquared += sumSquared;

}



void simulatePathsHalton(const Payoff& payoff, double S0, double r, double sigma, double T, int numPaths, int numSteps, int startIndex, const std::vector<int>& bases, ThreadTotals& out)
{
    double discount = std::exp(-r * T);
    double dt = T / numSteps;
    double sum = 0;
    double sumSquared = 0;
    std::vector<double> path(numSteps);
    for (int i = 0; i < numPaths; i++)
    {
        double S = S0;
        for (int step = 0; step < numSteps; step++)
        {
            double u = halton(startIndex + i, bases[step]) + pixelOffset(step, 0);
            if (u >= 1.0)
            {
                u -= 1.0;
            }

            double Z = inverseNormalCDF(u);
            S = S * std::exp((r - sigma * sigma / 2) * dt + sigma * std::sqrt(dt) * Z);
            path[step] = S;

        }
        double discounted = payoff.calculate(path) * discount;
        sum += discounted;
        sumSquared += discounted * discounted;
    }

    out.sum = sum;
    out.sumSquared += sumSquared;
}

MCResult monteCarloPrice(const Payoff& payoff, double S0, double r, double sigma, double T, int numPaths, int numSteps, int numThreads)
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

        threads.push_back(std::thread(simulatePathsGeneric, std::cref(payoff), S0, r, sigma, T, pathsForThisThread, numSteps, 42 + i, std::ref(totals[i])));
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

MCResult monteCarloPriceHalton(const Payoff& payoff, double S0, double r, double sigma, double T, int numPaths, int numSteps, int numThreads)
{
    std::vector<int> bases = firstPrimes(numSteps);
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

        int startIndex = 1 + i * pathsPerThread;
        threads.push_back(std::thread(simulatePathsHalton, std::cref(payoff), S0, r, sigma, T, pathsForThisThread, numSteps, startIndex, std::cref(bases), std::ref(totals[i])));
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

double impliedVol(double marketPrice, double S0, double K, double r, double T)
{
    double low = 0.01;
    double high = 3.0;
    double mid = 0;

    while ((high - low) > 1e-8)
    {
        mid = (low + high) / 2;
        if (blackScholesCall(S0, K, r, mid, T) > marketPrice)
        {
            high = mid;
        }
        else
        {
            low = mid;
        }
    }
    return mid;
}

void impliedVolTable(const std::string& filename, double S0, double r, double T)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cout << "Could not open " << filename << "\n";
        return;
    }
    std::string line;
    std::getline(file, line);              

    std::cout << std::setw(10) << "Strike" << std::setw(12) << "Mid" << std::setw(12) << "IV" << "\n";

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string field;

        std::getline(ss, field, ',');
        double strike = std::stod(field);

        std::getline(ss, field, ',');
        double bid = std::stod(field);

        std::getline(ss, field, ',');
        double ask = std::stod(field);

        double mid = (bid + ask) / 2;
        double iv = impliedVol(mid, S0, strike, r, T);
        std::cout << std::setw(10) << strike
            << std::setw(12) << mid
            << std::setw(12) << iv << "\n";

    }
}
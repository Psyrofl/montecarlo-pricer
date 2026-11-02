#pragma once
#include <algorithm>
#include <vector>

class Payoff
{
public:
    virtual ~Payoff() = default;
    virtual double calculate(const std::vector<double>& path) const = 0;
};

class VanillaPayoff : public Payoff
{
public:
    VanillaPayoff(double K) : K(K) {}

    double calculate(const std::vector<double>& path) const override
    {
        return std::max(path.back() - K, 0.0);
    }

private:
    double K;
};

class AsianPayoff : public Payoff
{
public:
    AsianPayoff(double K) : K(K) {}

    double calculate(const std::vector<double>& path) const override
    {
        double priceSum = 0;
        for (double S : path)
        {
            priceSum += S;
        }

        double average = priceSum / path.size();
        return std::max(average - K, 0.0);
    }

private:
    double K;

};

class BarrierPayoff : public Payoff
{
public:
    BarrierPayoff(double K, double barrier) : K(K), barrier(barrier) {}

    double calculate(const std::vector<double>& path) const override
    {
        double priceSum = 0;
        for (double S : path)
        {
            if (S >= barrier)
            {
                return 0.0;
            }
            else
            { 
                priceSum += S;
            }

        }

        return std::max(path.back() - K, 0.0);

    }

private:
    double K;
    double barrier;
};
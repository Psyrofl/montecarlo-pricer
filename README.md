# Monte Carlo Option Pricer

A multithreaded Monte Carlo engine in C++20 that prices European and path-dependent
options. Every result is checked against an exact answer, and the engine uses
variance reduction and quasi-random sampling to get accurate prices from fewer
simulations. It also calibrates to real S&P 500 option quotes.

## Features

- European, Asian, barrier and lookback calls, priced through one shared engine
- Black–Scholes formula as the exact reference for testing
- Standard errors and convergence measurement
- Multithreading with `std::thread`
- Variance reduction: antithetic variates and control variates
- Quasi-Monte Carlo with Halton sequences, including per-dimension offsets
- Implied volatility from real SPX quotes, with interpolation across strikes

## Results

All results use S0 = 100, K = 100, r = 5%, σ = 20% and T = 1 year, unless stated
otherwise. The exact Black–Scholes price is **10.4506**.

### Convergence

| Paths | Price | Standard error | Actual error |
|---|---|---|---|
| 1,000 | 11.4193 | 0.5030 | 0.9687 |
| 10,000 | 10.5951 | 0.1477 | 0.1445 |
| 100,000 | 10.4741 | 0.0467 | 0.0235 |
| 1,000,000 | 10.4682 | 0.0148 | 0.0176 |

The standard error falls by about √10 ≈ 3.2× for every 10× more paths, as expected
for Monte Carlo.

### Multithreading

10 million paths on an 8-core, 16-thread CPU:

| Threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Speedup | 1× | 2.0× | 3.8–3.9× | 5.1–6.0× | 5.7–9.4× |

The speedups vary between runs, especially at 16 threads, which is limited by
having only 8 physical cores. Each thread uses its own random seed and keeps its totals in local variables,
which avoids false sharing between threads.

### Variance reduction

10 million paths, with the variance reduction factor relative to standard Monte Carlo:

| Method | Price | Standard error | Variance reduction |
|---|---|---|---|
| Standard | 10.4499 | 0.00465 | 1× |
| Antithetic variates | 10.4500 | 0.00329 | 2.0× |
| Control variates | 10.4497 | 0.00177 | 6.9× |

The control variate uses the discounted final stock price, whose expected value is
exactly S0. Its coefficient β is estimated from a separate pilot run, so it doesn't
bias the main estimate.

### Path-dependent options

1 million paths with 252 daily steps:

| Option | Payoff | Price |
|---|---|---|
| European call | Final price − K | 10.45 |
| Asian call | Average price − K | 5.78 |
| Lookback call | Highest price − K | 18.35 |
| Up-and-out barrier at 150 | European payoff, or 0 if the price reaches 150 | 7.79 |
| Up-and-out barrier at 120 | European payoff, or 0 if the price reaches 120 | 1.32 |

With a single step, every one of these must equal the European price, and they do.
That test catches mistakes in each new payoff without needing a separate formula.

### Halton sequences vs random numbers

A Halton sequence spreads its points evenly, instead of clumping like random
numbers. Each time step uses a different prime base, and the inverse normal CDF
turns the evenly spread points into normal samples.

**European call, error against Black–Scholes:**

| Paths | Random error | Halton error |
|---|---|---|
| 1,000 | 0.9687 | 0.1206 |
| 10,000 | 0.1445 | 0.0178 |
| 100,000 | 0.0235 | 0.0024 |
| 1,000,000 | 0.0176 | 0.00028 |

Halton with 10,000 paths matches random sampling with 1,000,000: the same accuracy
from 100× fewer paths.

**Many time steps, 100,000 paths:** a European call built from many steps still has
an exact Black–Scholes price, but each path now needs one number per step.

| Steps | Random error | Plain Halton error | Halton with per-step offsets |
|---|---|---|---|
| 1 | 0.0468 | 0.0024 | 0.0024 |
| 12 | 0.0204 | 0.0175 | 0.0011 |
| 52 | 0.0244 | 0.1887 | 0.00017 |
| 252 | 0.0008 | 2.1755 | 0.0861 |

- **Plain Halton breaks in high dimensions.** Neighbouring large prime bases, such as
  1597 and 1601, produce almost identical early values, so the late steps move
  together, and the price ends up 2.18 off.
- **A fixed random offset for each step** (a Cranley–Patterson rotation) fixes this
  up to about 50 steps, but at 252 steps it is still worse than random sampling.
- The table shows one run of each, so the exact ratios include some luck.

The standard fix for many steps is a Brownian bridge, which builds the most
important parts of the path from the best Halton dimensions first.

### Real market data: implied volatility

SPX call quotes from Cboe on Nov 5, 2026, expiring Jan 8, 2027 (64 days), with
SPX at 7,688.62 and a 3-month Treasury yield of 4.13%. Each implied volatility is
found by bisection, searching for the σ where Black–Scholes matches the quote's mid
price.

| Strike | 7000 | 7200 | 7400 | 7600 | 7700 | 7800 | 8000 |
|---|---|---|---|---|---|---|---|
| Implied volatility | 22.3% | 19.9% | 17.6% | 15.4% | 14.3% | 13.5% | 12.1% |

If Black–Scholes described the market exactly, every strike would have the same
implied volatility. Instead, low strikes cost much more, which is the **volatility
skew**: markets tend to fall faster than they rise, so options that pay off in a
crash are more expensive. The 7700 value matches Cboe's own figure of 0.14.

**Pricing a strike that isn't in the data:** interpolating the skew to price the
7650 call, against its market mid price of 238.25:

| Volatility used | σ | Price | Error |
|---|---|---|---|
| Interpolated from the skew | 14.86% | 240.20 | +1.95 |
| Flat, from the nearest strike | 14.35% | 233.84 | −4.41 |

The Monte Carlo price with the skew volatility was 240.11 ± 0.32, matching
Black–Scholes. The 7650 quote was taken earlier than the others, while prices were
moving by about 1–2 points, which accounts for part of the remaining error.

## Design

The simulation engine is separate from the option types. Each option is a small
class that turns a simulated price path into a payoff:

```cpp
class Payoff
{
public:
    virtual ~Payoff() = default;
    virtual double calculate(const std::vector<double>& path) const = 0;
};
```

- The engine simulates the paths and handles the threading, and never needs to know
  which option it's pricing. Adding the lookback call took about 10 lines.
- `calculate` is `const`, so every thread can share one payoff object safely. The
  data that changes, the price path, lives in a separate vector for each thread.

## Shared with my ray tracer

This project and my [path tracer](https://github.com/Psyrofl/raytracer) are both
Monte Carlo methods, and they share code and ideas:

| Ray tracer | Pricer |
|---|---|
| Averages random light paths per pixel | Averages random price paths |
| Materials shared by threads, hit records per ray | Payoffs shared by threads, price paths per thread |
| Texture lookups between pixels | Volatility lookups between strikes |
| The Cornell box as a known test scene | Black–Scholes as a known test answer |

Both projects use the same `sampling.h`, with the Halton sequence, the inverse
normal CDF and the offset hash. In the ray tracer, using Halton for the position
inside each pixel gave no measurable improvement, because the light bounces still
use dozens of random numbers per path. Together, the two results show where
quasi-random sampling works: when it chooses all of a sample's numbers, and when
there are only a few of them.

## Building

**Windows:** open `montecarlo-pricer.slnx` in Visual Studio, select **Release | x64**
and run. The program reads `spx_calls.csv` from the `montecarlo-pricer` folder.

**macOS or Linux:**

```
cd montecarlo-pricer
clang++ -std=c++20 -O2 -Wall *.cpp -o pricer
./pricer
```

## What's next

- A Brownian bridge, so Halton works for paths with many steps
- Greeks: how the price changes with each input
- A second model, such as Heston, where volatility itself is random

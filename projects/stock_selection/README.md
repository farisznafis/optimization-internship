# Challenge 2: Stock selection

Allocate a USD 10,000 portfolio across seven stocks (A to G) to maximise the annualised average
return while keeping risk under control. The original brief is in [`brief.txt`](brief.txt) and
[`brief.png`](brief.png).

## Data

- [`data/returns.csv`](data/returns.csv): yearly returns 2018 to 2022 per stock, plus the team's
  spreadsheet columns (`average`, `aar`, ...).
- [`data/summary.csv`](data/summary.csv): per-stock summary (total and average return, risk).

The solver takes the annualised average return $a_i$ from the `aar` column and computes the risk
$r_i$ as the population standard deviation of the yearly returns (this reproduces the `risk` column
of `summary.csv`).

> **Note on AAR.** The spreadsheet computed `aar` as $(1 + \bar{R})^{1/5} - 1$ with $\bar{R}$ in
> percent units (for stock A, $\bar{R} = 3.2$ gives 33.2%). With $\bar{R}$ as a fraction (0.032) the
> AAR would be about 0.6%, and no portfolio could reach the 20% return target. The model keeps the
> team's values as given.

## Model

Variables: allocation $x_i \in [0, 1]$ and selection $y_i \in \{0, 1\}$.

$$
\begin{aligned}
\max\quad & \textstyle\sum_i a_i x_i \\
\text{s.t.}\quad & \textstyle\sqrt{\sum_i r_i^2 x_i^2} \le 15\% && \text{portfolio risk} \\
& \textstyle\sum_i a_i x_i \ge 20\% && \text{minimum return} \\
& \textstyle\sum_i y_i \ge 3 && \text{at least three stocks} \\
& 5.25\%\, y_i \le x_i \le y_i && \text{selected stocks get at least 5.25\%} \\
& \textstyle\sum_i x_i = 1 && \text{fully invested}
\end{aligned}
$$

The risk constraint is quadratic, so the model is a MIQCP solved with Gurobi. It is small enough for
the size-limited license that ships with `gurobipy`.

## Run

```bash
optim stock-selection
optim stock-selection --max-risk 0.10 --min-stocks 4 --budget 25000
```

The output lists each selected stock with its share, USD amount and return contribution, followed by
the portfolio's expected return and risk.

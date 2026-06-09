# CEO Salary: Inference and Prediction

**What drives CEO pay, firm size, or firm performance?** A cross-sectional study of 209 firms that pairs classical econometric inference with machine-learning prediction, and uses a Monte Carlo simulation to stress-test when each method can be trusted.

`Python` · `statsmodels` · `scikit-learn` · `OLS` · `KNN` · `Decision Trees` · `Random Forest` · `Monte Carlo simulation`

---

## TL;DR — key findings

- **Firm size dominates.** Across every method (OLS, KNN, decision trees, random forest), firm size (log-sales) is the single strongest determinant of CEO compensation. In the random forest it carries 61–82% of the predictive weight.
- **Performance matters at the threshold, not the margin.** A simple dummy for *whether returns were negative* (`NegROS`) even explains pay better than the continuous return-on-equity (`ROE`) measure, consistent with CEO compensation structure that punish losses more than they reward incremental gains.
- **KNN beats OLS out-of-sample.** Even after adding a quadratic size term to OLS, KNN achieves lower test MSE (best: 0.241 vs 0.30) by exploiting *local, asymmetric* structure that a global linear model averages away.
- **But OLS is more robust.** A 500 replications Monte Carlo study shows that once outliers are introduced, OLS degrades more gradually than KNN or trees, which is also a clean illustration of the bias–variance trade-off in small sample size.
- **Honest limits.** With only 209 observations, the random forest underperformed on the baseline specification, with only around 14% of salary variation are explained, pointing to the limitation of the given data (reputation, specific skills, or private negotiation).

---

## Why this project

Most workers are paid on fixed agreements, CEO pay is not. Because executives have scarce skills and bargaining power, firms tie their compensation to performance to align incentives with shareholders. This project asks **how much of that pay is actually explained by performance versus the sheer scale of the firm**, and treats the question from two angles that are usually taught separately:

- **Inference**: building regression models to test whether performance measures have a statistically meaningful relationship with pay, with full diagnostic testing.
- **Prediction**: applying flexible non-parametric methods (KNN, decision trees, random forest) to see whether they outperform OLS out-of-sample, and running a simulation study to understand when each approach can be trusted.

---

## Results at a glance

**Out-of-sample test MSE — OLS vs KNN across model specifications** (lower is better):

![OLS vs KNN test MSE](ceo-salary-inference-prediction/figures/ols_vs_knn_test_mse.png)

KNN is lower everywhere, and both methods collapse in quality when firm size is dropped (M8) — confirming size as the backbone predictor.

**Monte Carlo simulation: OLS is the most robust method across all outlier scenarios**

![Monte Carlo results](ceo-salary-inference-prediction/figures/monte_carlo_results.png)

**Correlation structure of the data:**

![Correlation heatmap](ceo-salary-inference-prediction/figures/correlation_heatmap.png)

`sales` and `lsales` are near-perfectly collinear (only one belongs in the model), and `salary`/`lsalary` correlate most strongly with firm size — visible before a single model is fit.

---

## Methods

| Approach | What it does here | Key techniques |
|---|---|---|
| **Linear regression (OLS)** | Baseline inference: build up from a simple log–log model to a full specification | log transforms, quadratic (centered) terms, dummy-trap-safe industry dummies, RESET / Breusch–Pagan / Jarque–Bera / Durbin–Watson / VIF diagnostics |
| **K-Nearest Neighbours** | Flexible prediction, both regression and classification (above/below median pay) | nested k-fold CV for tuning *k*, scaled vs unscaled features, partial dependence plots |
| **Decision tree + Random Forest** | Interpretable splits and an ensemble for variance reduction | cost-complexity pruning, GridSearchCV, min-samples-per-leaf constraints, feature importance |
| **Monte Carlo simulation** | Stress-test all three methods under outlier scenarios | custom data-generating process, Student-t errors, 500 repetitions, MSE comparison across 0% / 5% / 15% outliers |

A recurring methodological highlight documented in the report is the **"block trap"** — how unshuffled cross-validation folds (the data is ordered by industry) pushed the pruning algorithm toward a useless single-node tree, and how explicit shuffling + leaf constraints fixed it.

---

## Repository structure

```
.
├── data/
│   └── ceo_salary_1990.csv        # 209 firms, cross-section (course-provided)
├── notebooks/
│   ├── 01_linear_regression_and_knn.ipynb
│   ├── 02_decision_tree_and_random_forest.ipynb
│   └── 03_simulation_study.ipynb
├── report/
│   ├── CEO_Salary_Report.pdf       # full write-up with all results & diagnostics
│   └── Assignment_brief.pdf        # original case description
├── figures/                        # figures used in this README
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

A cross-section of **209 firms (1990)** with CEO `salary`, firm `sales`, performance measures (`roe`, `ros`, `pcroe`), and industry dummies (`indus`, `finance`, `consprod`, `utility`), plus log versions of salary and sales. No missing values. The dataset was provided as part of the course and is a classic teaching dataset for executive-compensation analysis, results should be read in the context of its age and modest size.


## Authors

Group project for **Data Science Practical (Year 2 - Bachelor)**, BSc Econometric and Data Science, School of Business and Economics, Vrije Universiteit Amsterdam.

- Tu Anh Nhu Vo
- Floris Drese
- Branca Hos

## License

MIT — see [LICENSE](LICENSE).

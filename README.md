# Strategic Classification and Behavioural Response

[![tests](https://github.com/MUmairSarwar/strategic-classification/actions/workflows/tests.yml/badge.svg)](https://github.com/MUmairSarwar/strategic-classification/actions/workflows/tests.yml)

I built a synthetic simulation to measure how a fixed logistic-regression model responds when rejected individuals can raise one feature to cross the decision boundary. I implemented the individual best response and measured the change in holdout performance and sensitivity to the cost of changing that feature.

![Experiment overview](reports/figures/strategic_classification.svg)

## Model

Everything is synthetic, with two features so the effect can be plotted:

$$x_1, x_2 \sim \mathcal{N}(0, 1), \qquad P(y = 1 \mid x) = \sigma(1.2\,x_1 + 0.8\,x_2 - 0.2),$$

with $n = 2500$ and a stratified 75/25 train/test split. A logistic regression $f(x) = w^\top x + b$ is trained on the unmodified training data and accepts when $f(x) \ge 0$.

On the test set, only $x_1$ can be changed. A rejected individual ($f(x) < 0$) needs

$$\delta = \frac{-f(x)}{w_1} + 10^{-6}$$

to reach the boundary, and makes that change only if

$$\delta \le \delta_{\max} = 2 \quad \text{and} \quad c\,\delta < B = 1,$$

where $c$ is the cost per unit of change and $B$ the benefit of acceptance. Accepted individuals do nothing, and nobody's label $y$ changes. This is pure gaming of the score, not real improvement. The model is not retrained after the response.

At the default cost $c = 0.4$ the budget condition only rules out $\delta \ge 2.5$, so the cap $\delta_{\max} = 2$ is the constraint that actually binds. The cost starts to matter for $c > 0.5$, which is the range covered by the sensitivity panel in the figure ($c$ from 0.1 to 1.2).

## Results

Test set of 625 individuals, default cost $c = 0.4$:

| | Before response | After response |
|---|---:|---:|
| Accuracy | 70.4% | 50.6% |
| ROC AUC | 0.800 | 0.773 |
| Share accepted | 46.1% | 96.3% |

50.2% of the test individuals changed $x_1$, by 0.84 on average. Almost everyone who was rejected could reach the boundary within the cap, so the classifier ends up accepting nearly everyone and its accuracy drops towards what accepting everyone would give. ROC AUC falls less: all manipulators end up just above the boundary, so the model loses the ordering among them, but they still rank below the originally accepted individuals and above those who could not reach the boundary.

These numbers follow from the assumptions above and are stored in [`reports/metrics.json`](reports/metrics.json). The cost sensitivity is only shown in the figure; it is not written to the metrics file.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m src.simulate   # rewrites reports/metrics.json and the figure
```

The code is a single module, [`src/simulate.py`](src/simulate.py). The tests check that data generation is reproducible, that the response only increases $x_1$ and leaves $x_2$ alone, and that the metrics are in range.

## Limitations and possible extensions

- Everyone has the same cost and benefit. Heterogeneous costs would let the response differ between groups, which is where fairness questions start.
- Only one feature is manipulable and the change is pure gaming. Separating gaming from genuine improvement needs a causal model of how features affect the label.
- The classifier does not anticipate the response. A natural next step is to train a model that expects best responses, as in the strategic-classification framework of Hardt, Megiddo, Papadimitriou and Wootters (ITCS 2016), and compare it with this naive baseline.
- One data seed and one split are used.

## History

The [initial simulation](https://github.com/MUmairSarwar/strategic-classification/commit/66103c3) was committed on 6 February 2026. In August 2026 I implemented the individual best-response model and added the sensitivity plot, tests and recorded outputs. The results above come from that version. The data is generated from a fixed seed in the simulation, so there is no external dataset.

## Author

Muhammad Umair Sarwar, M.Sc. Mathematics (Mathematics in Data Science), TU Darmstadt. Other projects are listed on my [GitHub profile](https://github.com/MUmairSarwar).

Licence: MIT, see [LICENSE](LICENSE).

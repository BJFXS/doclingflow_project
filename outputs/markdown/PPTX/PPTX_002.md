# Evaluation Metrics CS229

Yining Chen

(Adapted from slides by Anand Avati)

May 1, 2020

# Topics

- Why are metrics important?
- Binary classifiers
- Rank view, Thresholding
- Metrics
- Confusion Matr .... IX
- Point metrics: Accuracy, Precision, Recall / Sensitivity, Specificity, F-score
- Summary metrics: AU-ROC, AU-PRC, Log-loss.
- Choosing Metrics
- Class Imbalance
- Failure scenarios for each metr .... IC
- Multi-class

# Why are metrics important?

- Training objective (cost function) is only a proxy for real world objectives.
- Metrics help capture a business goal into a quantitative target (not all errors are equal).
- Helps organize ML team effort towards that target.
- Generally in the form of improving that metric on the dev set.
- Useful to quantify the “gap” between:
- Desired performance and baseline (estimate effort initially).
- Desired performance and current performance.
- Measure progress over time.
- Useful for lower level tasks and debugging (e.g. diagnosing bias vs variance).
- Ideally training objective should be the metric, but not always possible. Still, metrics are useful and important for evaluation.

# Binary Classification

- x is input
- y is binary output (0/1)
- Model is ŷ = h(x)
- Two types of models
- Models that output a categorical class directly (K-nearest neighbor, Decision tree)
- Models that output a real valued score (SVM, Logistic Regression)
- Score could be margin (SVM), probability (LR, NN)
- Need to pick a threshold
- We focus on this type (the other type can be interpreted as an instance)

# Score based models

Score = 1

Score = 0

<table>
  <thead>
    <tr>
      <th></th>
      <th>Positive example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td>Negative example</td>
    </tr>
  </tbody>
</table>

Prevalence =

| # positive examples                         |
|---------------------------------------------|
| # positive examples +  # negatives examples |

Example of Score: Output of logistic regression.

For most metrics: Only ranking matters.

If too many examples: Plot class-wise histogram.

# Threshold -&gt; Classifier -&gt; Point Metrics

Th=0.5

|   Th |
|------|
|  0.5 |

     Label positive                         Label negative

    Predict Negative                      Predict Positive

# Point metrics: Confusion Matrix

     Label Positive                         Label Negative

    Predict Negative                      Predict Positive

9

8

2

1

Th=0.5

|   Th |
|------|
|  0.5 |

Properties:

- Total sum is fixed (population).
- Column sums are fixed (class-wise population).
- Quality of model &amp; threshold decide how columns are split into rows.
- We want diagonals to be “heavy”, off diagonals to be “light”.

# Point metrics: True Positives

#

     Label positive                         Label negative

9

8

2

1

Th=0.5

    Predict Negative                      Predict Positive

|   Th |
|------|
|  0.5 |

|   TP |
|------|
|    9 |

# Point metrics: True Negatives

#

     Label positive                         Label negative

9

8

2

1

Th=0.5

|   Th |
|------|
|  0.5 |

|   TP |
|------|
|    9 |

|   TN |
|------|
|    8 |

    Predict Negative                      Predict Positive

# Point metrics: False Positives

#

     Label positive                         Label negative

9

8

2

1

Th=0.5

|   Th |
|------|
|  0.5 |

|   TP |
|------|
|    9 |

|   TN |
|------|
|    8 |

    Predict Negative                      Predict Positive

|   FP |
|------|
|    2 |

# Point metrics: False Negatives

#

     Label positive                         Label negative

9

8

2

1

Th=0.5

|   Th |
|------|
|  0.5 |

|   TP |
|------|
|    9 |

|   TN |
|------|
|    8 |

    Predict Negative                      Predict Positive

|   FP |
|------|
|    2 |

|   FN |
|------|
|    1 |

# FP and FN also called Type-1 and Type-2 errors

![Image](pptx2/document_artifacts/image_000000_37cfd7963bfee03e6b66ae0f9b1b14de64575e7e55805f3f0550146fb31c9066.png)

Could not find true source of image to cite

# Point metrics: Accuracy

#

     Label positive                         Label negative

9

8

2

1

Th=0.5

|   Th |
|------|
|  0.5 |

|   TP |
|------|
|    9 |

|   TN |
|------|
|    8 |

    Predict Negative                      Predict Positive

|   FP |
|------|
|    2 |

|   FN |
|------|
|    1 |

|   Acc |
|-------|
|   .85 |

Equivalent to 0-1 Loss!

# Point metrics: Precision

#

     Label positive                         Label negative

9

8

2

1

Th=0.5

|   Th |
|------|
|  0.5 |

|   TP |
|------|
|    9 |

|   TN |
|------|
|    8 |

    Predict Negative                      Predict Positive

|   FP |
|------|
|    2 |

|   FN |
|------|
|    1 |

|   Acc |
|-------|
|   .85 |

|   Pr |
|------|
|  .81 |

# Point metrics: Positive Recall (Sensitivity)

#

     Label positive                         Label negative

9

8

2

1

Th=0.5

|   Th |
|------|
|  0.5 |

|   TP |
|------|
|    9 |

|   TN |
|------|
|    8 |

    Predict Negative                      Predict Positive

|   FP |
|------|
|    2 |

|   FN |
|------|
|    1 |

|   Acc |
|-------|
|   .85 |

|   Pr |
|------|
|  .81 |

|   Recall |
|----------|
|       .9 |

Trivial 100% recall = pull everybody above the threshold.

$$Trivial 100% precision = push everybody below the threshold except 1 green on top.$$

(Hopefully no gray above it!)

Striving for good precision with 100% recall =

pulling up the lowest green as high as possible in the ranking.

Striving for good recall with 100% precision =

pushing down the top gray as low as possible in the ranking.

# Point metrics: Negative Recall (Specificity)

#

     Label positive                         Label negative

9

8

2

1

Th=0.5

|   Th |
|------|
|  0.5 |

|   TP |
|------|
|    9 |

|   TN |
|------|
|    8 |

    Predict Negative                      Predict Positive

|   FP |
|------|
|    2 |

|   FN |
|------|
|    1 |

|   Acc |
|-------|
|   .85 |

|   Pr |
|------|
|  .81 |

|   Recall |
|----------|
|       .9 |

|   Spec |
|--------|
|    0.8 |

# Point metrics: F1-score

#

     Label positive                         Label negative

9

8

2

1

Th=0.5

|   Th |
|------|
|  0.5 |

|   TP |
|------|
|    9 |

|   TN |
|------|
|    8 |

    Predict Negative                      Predict Positive

|   FP |
|------|
|    2 |

|   FN |
|------|
|    1 |

|   Acc |
|-------|
|   .85 |

|   Pr |
|------|
|  .81 |

|   Recall |
|----------|
|       .9 |

|   Spec |
|--------|
|     .8 |

|   F1 |
|------|
| .857 |

![Image](pptx2/document_artifacts/image_000001_0af10ecb20ec3c42990c15df24502bc86405c634e4772e17ac67a950e1051479.png)

# Point metrics: Changing threshold

#

     Label positive                         Label negative

7

8

2

3

Th=0.6

|   Th |
|------|
|  0.6 |

|   TP |
|------|
|    7 |

|   TN |
|------|
|    8 |

    Predict Negative                      Predict Positive

|   FP |
|------|
|    2 |

|   FN |
|------|
|    3 |

|   Acc |
|-------|
|   .75 |

|   Pr |
|------|
|  .77 |

|   Recall |
|----------|
|       .7 |

|   Spec |
|--------|
|     .8 |

|   F1 |
|------|
| .733 |

# effective thresholds = # examples + 1

<table>
  <thead>
    <tr>
      <th>Threshold</th>
      <th>TP</th>
      <th>TN</th>
      <th>FP</th>
      <th>FN</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Specificity</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.00</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>10</td>
      <td>0.50</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>0.95</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>9</td>
      <td>0.55</td>
      <td>1</td>
      <td>0.1</td>
      <td>1</td>
      <td>0.182</td>
    </tr>
    <tr>
      <td>0.90</td>
      <td>2</td>
      <td>10</td>
      <td>0</td>
      <td>8</td>
      <td>0.60</td>
      <td>1</td>
      <td>0.2</td>
      <td>1</td>
      <td>0.333</td>
    </tr>
    <tr>
      <td>0.85</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>8</td>
      <td>0.55</td>
      <td>0.667</td>
      <td>0.2</td>
      <td>0.9</td>
      <td>0.308</td>
    </tr>
    <tr>
      <td>0.80</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>7</td>
      <td>0.60</td>
      <td>0.750</td>
      <td>0.3</td>
      <td>0.9</td>
      <td>0.429</td>
    </tr>
    <tr>
      <td>0.75</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
      <td>6</td>
      <td>0.65</td>
      <td>0.800</td>
      <td>0.4</td>
      <td>0.9</td>
      <td>0.533</td>
    </tr>
    <tr>
      <td>0.70</td>
      <td>5</td>
      <td>9</td>
      <td>1</td>
      <td>5</td>
      <td>0.70</td>
      <td>0.833</td>
      <td>0.5</td>
      <td>0.9</td>
      <td>0.625</td>
    </tr>
    <tr>
      <td>0.65</td>
      <td>5</td>
      <td>8</td>
      <td>2</td>
      <td>5</td>
      <td>0.65</td>
      <td>0.714</td>
      <td>0.5</td>
      <td>0.8</td>
      <td>0.588</td>
    </tr>
    <tr>
      <td>0.60</td>
      <td>6</td>
      <td>8</td>
      <td>2</td>
      <td>4</td>
      <td>0.70</td>
      <td>0.750</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.667</td>
    </tr>
    <tr>
      <td>0.55</td>
      <td>7</td>
      <td>8</td>
      <td>2</td>
      <td>3</td>
      <td>0.75</td>
      <td>0.778</td>
      <td>0.7</td>
      <td>0.8</td>
      <td>0.737</td>
    </tr>
    <tr>
      <td>0.50</td>
      <td>8</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>0.80</td>
      <td>0.800</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.800</td>
    </tr>
    <tr>
      <td>0.45</td>
      <td>9</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>0.85</td>
      <td>0.818</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.857</td>
    </tr>
    <tr>
      <td>0.40</td>
      <td>9</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>0.80</td>
      <td>0.750</td>
      <td>0.9</td>
      <td>0.7</td>
      <td>0.818</td>
    </tr>
    <tr>
      <td>0.35</td>
      <td>9</td>
      <td>6</td>
      <td>4</td>
      <td>1</td>
      <td>0.75</td>
      <td>0.692</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.783</td>
    </tr>
    <tr>
      <td>0.30</td>
      <td>9</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>0.70</td>
      <td>0.643</td>
      <td>0.9</td>
      <td>0.5</td>
      <td>0.750</td>
    </tr>
    <tr>
      <td>0.25</td>
      <td>9</td>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>0.65</td>
      <td>0.600</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>0.720</td>
    </tr>
    <tr>
      <td>0.20</td>
      <td>9</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>0.60</td>
      <td>0.562</td>
      <td>0.9</td>
      <td>0.3</td>
      <td>0.692</td>
    </tr>
    <tr>
      <td>0.15</td>
      <td>9</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>0.55</td>
      <td>0.529</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.667</td>
    </tr>
    <tr>
      <td>0.10</td>
      <td>9</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>0.50</td>
      <td>0.500</td>
      <td>0.9</td>
      <td>0.1</td>
      <td>0.643</td>
    </tr>
    <tr>
      <td>0.05</td>
      <td>10</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>0.55</td>
      <td>0.526</td>
      <td>1</td>
      <td>0.1</td>
      <td>0.690</td>
    </tr>
    <tr>
      <td>0.00</td>
      <td>10</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0.50</td>
      <td>0.500</td>
      <td>1</td>
      <td>0</td>
      <td>0.667</td>
    </tr>
  </tbody>
</table>

Score = 1

Score = 0

Threshold = 0.00

Threshold = 1.00

Threshold Scanning

# Summary metrics: Rotated ROC (Sen vs. Spec)

![Image](pptx2/document_artifacts/image_000002_a0bbd2a12b800d57bae539d80f37168b33a254df63287117f4c0c5ecf02bf736.png)

Score = 1

Score = 0

Sensitivity = True Pos / Pos

Specificity

= True Neg / Neg

Pos examples

Neg examples

Random Guessing

AUROC = Area Under ROC

= Prob[Random Pos ranked

higher than random Neg]

Agnostic to prevalence!

# Summary metrics: PRC (Recall vs. Precision)

![Image](pptx2/document_artifacts/image_000003_e31d0df6615407d2a779780c88272d84b0b95e875b8c12eb50c978ac1ba57c3a.png)

Score = 1

Score = 0

Recall = Sensitivity = True Pos / Pos

Precision

= True Pos /

Predicted Pos

Pos examples

Neg examples

AUPRC = Area Under PRC

= Expected precision for

Random threshold

$$Precision &gt;= prevalence$$

# Summary metrics:

Score = 1

Score = 0

Score = 1

Score = 0

Two models scoring the same data set. Is one of them better than the other?

Model A

Model B

# Summary metrics: Log-Loss vs Brier Score

- Same ranking, and therefore the same AUROC, AUPRC, accuracy!
- Rewards confident correct answers, heavily penalizes confident wrong answers.
- One perfectly confident wrong prediction is fatal.

-&gt; Well-calibrated model

- Proper scoring rule: Minimized at

Score = 1

Score = 0

Score = 1

Score = 0

![Image](pptx2/document_artifacts/image_000004_fce9632cd063f4377b42026dd183c0fd24f3ecacfb2f8a2afaf5abbbce28b27a.png)

![Image](pptx2/document_artifacts/image_000005_f9aad6ed30594820e349dafac949999acfba660a1e9699d7c784917b30fee9b6.png)

![Image](pptx2/document_artifacts/image_000006_a40cbfe272192a777ea93ebaf1b9e46f8a83708834265e5ef0246372c7ed116d.png)

# Calibration vs Discriminative Power

![Image](pptx2/document_artifacts/image_000007_64bb9e3931c6ec910e433987b17ad40f4d6638b965ee99aeda7f1d0112e5a288.png)

$$Logistic (th=0.5):$$

  Precision: 0.872

  Recall: 0.851

  F1: 0.862

  Brier: 0.099

$$SVC (th=0.5):$$

  Precision: 0.872

  Recall: 0.852

  F1: 0.862

  Brier: 0.163

Output

Fraction of Positives

Histogram

# Unsupervised Learning

- Log P(x) is a measure of fit in Probabilistic models (GMM, Factor Analysis)
- High log P(x) on training set, but low log P(x) on test set is a measure of overfitting
- Raw value of log P(x) hard to interpret in isolation

- K-means is trickier (because of fixed covariance assumption)

# Class Imbalance

Symptom: Prevalence &lt; 5% (no strict definition)

Metrics: May not be meaningful.

Learning: May not focus on minority class examples at all

(majority class can overwhelm logistic regression, to a lesser extent SVM)

# What happen to the metrics under class imbalance?

Accuracy: Blindly predicts majority class -&gt; prevalence is the baseline.

Log-Loss: Majority class can dominate the loss.

AUROC: Easy to keep AUC high by scoring most negatives very low.

AUPRC: Somewhat more robust than AUROC. But other challenges.

In general:     Accuracy  &lt; AUROC  &lt; AUPRC

Score = 1

Score = 0

1%

1%

98%

Rotated ROC

Specificity

= True Neg / Neg

Sensitivity = True Pos / Pos

“Fraudulent”

AUC = 98/99

# Multi-class

- Confusion matrix will be N * N (still want heavy diagonals, light off-diagonals)
- Most metrics (except accuracy) generally analyzed as multiple 1-vs-many
- Multiclass variants of AUROC and AUPRC (micro vs macro averaging)
- Class imbalance is common (both in absolute and relative sense)
- Cost sensitive learning techniques (also helps in binary Imbalance)
- Assign weights for each block in the confusion matrix.
- Incorporate weights into the loss function.

# Choosing Metrics

Some common patterns:

- High precision is hard constraint, do best recall (search engine results, grammar correction): Intolerant to FP
- Metric:

$$Recall at Precision = XX %$$

- High recall is hard constraint, do best precision (medical diagnosis): Intolerant to FN
- Metric:

$$Precision at Recall = 100 %$$

- Capacity constrained (by K)
- Metric: Precision in top-K.
- ……

# Thank You!

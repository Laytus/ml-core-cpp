# Logistic Regression

This document covers the classification foundations used in ML Core.

The goal is to understand logistic regression as a serious multivariate classification model, not as a scalar toy example.

Logistic regression is the first major classification model in ML Core because it connects directly to:

```txt
linear models
logits
probabilities
sigmoid activation
binary cross-entropy
decision thresholds
decision boundaries
regularization
evaluation metrics
the bridge toward neural networks
```

---

## 1. Multivariate Logistic Regression

Logistic regression is a supervised learning model for binary classification.

It predicts the probability that a sample belongs to the positive class.

For one sample with `n` features:

$$
x =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

and weights:

$$
w =
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n
\end{bmatrix}
$$

the model first computes a linear score:

$$
z = w^\top x + b
$$

where:

```txt
z = logit / raw linear score
w = weight vector
x = feature vector
b = bias / intercept
```

Then the score is passed through the sigmoid function:

$$
\hat{p} = \sigma(z)
$$

where:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The output:

```txt
p̂ = predicted probability of the positive class
```

So the full model is:

$$
\hat{p} = \sigma(w^\top x + b)
$$

---

### Binary target convention

For binary classification, targets are represented as:

```txt
y = 0:
    negative class

y = 1:
    positive class
```

The model predicts:

```txt
P(y = 1 | x)
```

That means:

```txt
p̂ close to 1:
    model believes the sample is likely positive

p̂ close to 0:
    model believes the sample is likely negative

p̂ close to 0.5:
    model is uncertain under the current parameters
```

---

### Dataset-level representation

ML Core should implement logistic regression in multivariate, dataset-level form.

Given:

```txt
X: m x n feature matrix
y: m-dimensional target vector with values 0 or 1
w: n-dimensional weight vector
b: scalar bias
```

where:

```txt
m = number of samples
n = number of features
```

the model computes logits for all samples:

$$
z = Xw + b
$$

Then probabilities:

$$
\hat{p} = \sigma(z)
$$

where the sigmoid is applied element-wise.

So:

$$
\hat{p}_i = \sigma(x_i^\top w + b)
$$

for each sample `i`.

---

### Logistic regression as a linear classifier

Although logistic regression outputs probabilities, its decision boundary is linear in feature space.

The class boundary occurs where:

$$
\hat{p} = 0.5
$$

Since:

$$
\sigma(0) = 0.5
$$

the decision boundary is:

$$
w^\top x + b = 0
$$

This means logistic regression is a linear classifier.

It can separate classes with a line, plane, or hyperplane depending on feature dimension.

```txt
2 features:
    decision boundary is a line

3 features:
    decision boundary is a plane

n features:
    decision boundary is a hyperplane
```

---

### What the model learns

Training logistic regression means finding:

```txt
weights w
bias b
```

so that predicted probabilities match the binary targets.

Each weight controls how one feature changes the logit.

A positive weight means:

```txt
increasing the feature increases the logit
increasing the logit increases the predicted probability
```

A negative weight means:

```txt
increasing the feature decreases the logit
decreasing the logit decreases the predicted probability
```

A weight near zero means:

```txt
the feature has little linear effect on the classification score
```

The bias shifts the decision boundary.

---

### Why logistic regression matters

Logistic regression is important because it introduces classification while preserving the linear-model structure from Phase 3.

It is also a bridge toward neural networks:

```txt
linear transformation:
    z = Xw + b

activation:
    p = sigmoid(z)

loss:
    binary cross-entropy

gradient-based training:
    update parameters using derivatives
```

This pattern is a simplified version of what later appears in neural networks.

---

## 2. Logits and Sigmoid Interpretation

A logit is the raw linear score before applying the sigmoid function.

For one sample:

$$
z = w^\top x + b
$$

The logit can be any real number:

```txt
z can be negative
z can be zero
z can be positive
```

The sigmoid maps that real number into a probability-like value between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

So:

$$
\hat{p} = \sigma(z)
$$

---

### Sigmoid behavior

Important sigmoid values:

```txt
z = 0:
    sigmoid(z) = 0.5

z >> 0:
    sigmoid(z) approaches 1

z << 0:
    sigmoid(z) approaches 0
```

Examples:

```txt
large positive logit:
    strong positive-class prediction

large negative logit:
    strong negative-class prediction

logit near zero:
    uncertain prediction
```

The sigmoid is monotonic:

```txt
larger z always means larger predicted probability
```

---

### Logits as log-odds

The logit has a probabilistic interpretation as log-odds.

If:

$$
p = P(y = 1 | x)
$$

then the odds are:

$$
\frac{p}{1-p}
$$

and the log-odds are:

$$
\log\left(\frac{p}{1-p}\right)
$$

Logistic regression models the log-odds as a linear function of the features:

$$
\log\left(\frac{p}{1-p}\right) = w^\top x + b
$$

This is why the raw score is called a logit.

---

### Why not use a linear output directly?

For regression, the model output can be any real number:

$$
\hat{y} = Xw + b
$$

For binary classification, we want something interpretable as a probability:

```txt
between 0 and 1
```

A linear output does not satisfy this.

It can produce:

```txt
negative values
values greater than 1
```

The sigmoid fixes this by mapping:

```txt
(-∞, +∞) → (0, 1)
```

---

### Numerical stability note

In implementation, sigmoid should be computed carefully for large positive or negative logits.

The direct expression:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

can overflow or underflow for very large values.

A stable implementation should handle large logits safely.

For Phase 4, ML Core should implement sigmoid in a way that avoids obvious numerical issues.

---

### Sigmoid derivative

The derivative of sigmoid is:

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

This derivative is important for gradient-based training.

However, when sigmoid is combined with binary cross-entropy, the final gradient simplifies nicely.

For logistic regression with binary cross-entropy, the gradient with respect to the weights becomes:

$$
\nabla_w J = \frac{1}{m}X^\top(\hat{p} - y)
$$

This simplicity is one reason sigmoid + binary cross-entropy is the standard pairing.

---

## 3. Binary Cross-Entropy in Vectorized Form

Logistic regression is not trained with MSE by default.

Instead, it uses Binary Cross-Entropy, also called log loss.

For one sample:

$$
L_i = -\left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]
$$

where:

```txt
y_i = true label, either 0 or 1
p̂_i = predicted probability for positive class
```

For the full dataset:

$$
BCE = -\frac{1}{m}
\sum_{i=1}^{m}
\left[
y_i \log(\hat{p}_i)
+
(1-y_i)\log(1-\hat{p}_i)
\right]
$$

---

### Intuition

Binary cross-entropy rewards confident correct predictions and penalizes confident wrong predictions.

If:

```txt
y = 1
```

then the loss becomes:

$$
-\log(\hat{p})
$$

So:

```txt
p̂ close to 1:
    small loss

p̂ close to 0:
    very large loss
```

If:

```txt
y = 0
```

then the loss becomes:

$$
-\log(1-\hat{p})
$$

So:

```txt
p̂ close to 0:
    small loss

p̂ close to 1:
    very large loss
```

This is exactly what we want for binary classification.

---

### Vectorized form

Let:

```txt
p̂: m-dimensional probability vector
y: m-dimensional target vector
```

Then binary cross-entropy can be written as:

$$
BCE =
-\frac{1}{m}
\left[
y^\top \log(\hat{p})
+
(1-y)^\top \log(1-\hat{p})
\right]
$$

where the logarithm is applied element-wise.

In implementation terms:

```cpp
loss = -mean(
    y * log(probabilities)
    + (1 - y) * log(1 - probabilities)
);
```

Using Eigen arrays, this should be implemented as element-wise operations.

---

### Probability clipping

Binary cross-entropy contains:

```txt
log(p̂)
log(1 - p̂)
```

If `p̂` is exactly `0` or exactly `1`, then the log becomes undefined:

```txt
log(0) = -∞
```

To avoid numerical issues, probabilities are usually clipped:

```txt
epsilon <= p̂ <= 1 - epsilon
```

For example:

```txt
epsilon = 1e-15
```

Then:

```txt
p̂_clipped = min(max(p̂, epsilon), 1 - epsilon)
```

This makes the loss finite.

ML Core should use probability clipping inside BCE.

---

### BCE gradient for logistic regression

For logistic regression:

$$
z = Xw + b
$$

$$
\hat{p} = \sigma(z)
$$

With binary cross-entropy, the gradient simplifies to:

$$
\nabla_w BCE =
\frac{1}{m}X^\top(\hat{p} - y)
$$

and:

$$
\frac{\partial BCE}{\partial b} =
\frac{1}{m}\sum_{i=1}^{m}(\hat{p}_i - y_i)
$$

This is very similar to linear regression, but with:

```txt
predictions = probabilities
residual-like error = probabilities - targets
```

For logistic regression:

```txt
p̂ - y
```

acts like the classification residual.

---

### Why not MSE for logistic regression?

MSE can be used mathematically, but it is not the standard choice for logistic regression.

Binary cross-entropy is preferred because:

```txt
it matches the Bernoulli likelihood interpretation
it strongly penalizes confident wrong predictions
it produces a clean gradient with sigmoid
it leads to a convex objective for logistic regression
```

This gives logistic regression a strong probabilistic and optimization foundation.

---

## 4. Thresholding and Decision Boundaries

Logistic regression outputs probabilities, not classes directly.

The probability output is:

$$
\hat{p} = P(y = 1 | x)
$$

To convert probabilities into class predictions, we apply a threshold.

The most common threshold is:

$$
t = 0.5
$$

Prediction rule:

$$
\hat{y}_{class} =
\begin{cases}
1 & \text{if } \hat{p} \ge t \\
0 & \text{if } \hat{p} < t
\end{cases}
$$

---

### Default threshold

With threshold `0.5`:

```txt
p̂ >= 0.5:
    predict positive class

p̂ < 0.5:
    predict negative class
```

Since:

$$
\sigma(0) = 0.5
$$

this is equivalent to:

```txt
z >= 0:
    predict positive class

z < 0:
    predict negative class
```

where:

$$
z = w^\top x + b
$$

So the default decision boundary is:

$$
w^\top x + b = 0
$$

---

### Decision boundary interpretation

The decision boundary is the set of points where the model is exactly uncertain at the selected threshold.

For threshold `0.5`, this is:

$$
w^\top x + b = 0
$$

For two features:

$$
w_1x_1 + w_2x_2 + b = 0
$$

This is a line.

Solving for `x_2`:

$$
x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}
$$

when:

$$
w_2 \ne 0
$$

This line separates the feature space into two predicted regions.

---

### Threshold variation

The threshold does not have to be `0.5`.

Changing the threshold changes the trade-off between false positives and false negatives.

Lower threshold:

```txt
more positive predictions
higher recall for positive class
possibly more false positives
```

Higher threshold:

```txt
fewer positive predictions
possibly higher precision
possibly more false negatives
```

So threshold choice depends on the problem.

Example:

```txt
medical screening:
    false negatives may be very costly
    lower threshold may be preferred

spam detection:
    false positives may be annoying
    threshold may be tuned carefully
```

ML Core should expose thresholding explicitly instead of hiding it inside the model.

---

### Classification metrics depend on threshold

Probability metrics and class metrics are different.

Binary cross-entropy evaluates probability quality.

Class metrics evaluate thresholded predictions.

Examples of threshold-dependent metrics:

```txt
accuracy
precision
recall
F1 score
confusion matrix
```

Changing the threshold can change these metrics without changing the underlying probabilities.

Therefore Phase 4 experiments should include threshold variation.

---

### ML Core implementation decision

The logistic regression model should support:

```txt
logits(X)
predict_proba(X)
predict_classes(X, threshold)
```

where:

```txt
logits:
    raw linear scores

predict_proba:
    sigmoid(logits)

predict_classes:
    thresholded class predictions
```

The threshold should be an explicit parameter.

Do not hardcode threshold behavior in a way that makes later analysis difficult.

---

## 5. Regularization in Classification

Regularization in logistic regression has the same purpose as in linear regression:

```txt
reduce overfitting
control weight magnitude
improve generalization
improve stability when features are correlated
```

The unregularized binary cross-entropy objective is:

$$
BCE(w,b)
$$

With Ridge regularization, the objective becomes:

$$
J(w,b) =
BCE(w,b) + \lambda \|w\|_2^2
$$

where:

```txt
λ = regularization strength
w = feature weights
b = bias
```

As in linear regression, ML Core should regularize weights but not the bias.

---

### Ridge logistic regression

The Ridge penalty is:

$$
\lambda \|w\|_2^2 =
\lambda w^\top w
$$

The gradient contribution is:

$$
2\lambda w
$$

So for logistic regression with Ridge:

$$
\nabla_w J =
\frac{1}{m}X^\top(\hat{p} - y)
+
2\lambda w
$$

and:

$$
\frac{\partial J}{\partial b} =
\frac{1}{m}\sum_{i=1}^{m}(\hat{p}_i - y_i)
$$

The bias gradient is unchanged because the bias is not regularized.

---

### Why regularization matters for classification

Without regularization, logistic regression can produce very large weights when classes are nearly linearly separable.

Large weights create very confident probabilities:

```txt
p̂ very close to 0
p̂ very close to 1
```

This may reduce training loss but can hurt generalization.

Ridge discourages extreme weights and can produce better-calibrated probability behavior.

---

### Scaling and regularization

As with Ridge linear regression, regularized logistic regression is sensitive to feature scale.

If features have very different scales, the weight penalty may affect features unevenly.

A leakage-safe workflow is:

```txt
1. split data
2. fit scaler on training features
3. transform train/validation/test
4. train logistic regression with regularization
5. evaluate probabilities and classes
```

The scaler must be fitted only on training data.

---

### Lasso conceptually

Lasso logistic regression uses an L1 penalty:

$$
J(w,b) =
BCE(w,b) + \lambda \|w\|_1
$$

where:

$$
\|w\|_1 = \sum_{j=1}^{n}|w_j|
$$

Like in linear regression, Lasso can encourage sparse weights.

This means some features may effectively be removed from the classifier.

However, Lasso is harder to optimize because the absolute value penalty is not differentiable at zero.

For Phase 4:

```txt
Ridge:
    implement

Lasso:
    document conceptually
    do not implement yet
```

---

### ML Core implementation decision

For Phase 4, logistic regression should support:

```txt
no regularization
Ridge regularization
explicit rejection of Lasso fitting for now
```

This should mirror the Phase 3 linear regression design.

Rules:

```txt
lambda = 0:
    same as unregularized logistic regression

lambda > 0 with Ridge:
    add L2 penalty to loss and weight gradient

bias:
    never regularized

Lasso:
    accepted conceptually in RegularizationConfig
    rejected during fit until implemented
```

This keeps the classification model aligned with the existing linear-model regularization design.

---

## 6. Softmax Regression and Multiclass Classification

Binary logistic regression handles classification with two classes:

```txt
negative class: 0
positive class: 1
```

It produces one probability:

$$
\hat{p} = P(y = 1 \mid x)
$$

The probability of the negative class is then:

$$
P(y = 0 \mid x) = 1 - \hat{p}
$$

For multiclass classification, there are more than two possible classes.

Example:

```txt
class 0
class 1
class 2
...
class K - 1
```

In that setting, the model should output one probability per class.

For `K` classes:

$$
\hat{p} =
\begin{bmatrix}
P(y = 0 \mid x) \\
P(y = 1 \mid x) \\
\vdots \\
P(y = K-1 \mid x)
\end{bmatrix}
$$

These probabilities must satisfy:

```txt
each probability is between 0 and 1
all probabilities sum to 1
```

Softmax regression is the multiclass extension of logistic regression.

It keeps the same linear-model structure, but instead of learning one weight vector and one bias, it learns one score function per class.

---

### From sigmoid to softmax

Binary logistic regression uses the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The sigmoid maps one logit to one probability.

Softmax regression uses the softmax function.

For a vector of logits:

$$
z =
\begin{bmatrix}
z_1 \\
z_2 \\
\vdots \\
z_K
\end{bmatrix}
$$

softmax produces:

$$
\hat{p}_k =
\frac{e^{z_k}}
{\sum_{j=1}^{K}e^{z_j}}
$$

for each class `k`.

So the output vector is:

$$
\hat{p} = softmax(z)
$$

The softmax turns raw class scores into a probability distribution.

Each class probability depends on every class logit because all logits appear in the denominator.

That means increasing one class logit increases that class probability while decreasing the relative probabilities of the other classes.

---

### Multiclass linear model

For binary logistic regression, the logit is:

$$
z = w^\top x + b
$$

For softmax regression, each class has its own linear score.

Using a weight matrix:

$$
W
$$

and a bias vector:

$$
b
$$

the logits for one sample are:

$$
z = W^\top x + b
$$

For a dataset:

```txt
X: m x n
W: n x K
b: K-dimensional
```

the vectorized logits are:

$$
Z = XW + b
$$

where:

```txt
Z: m x K
```

Each row contains the class logits for one sample.

Then softmax is applied row-wise:

$$
\hat{P} = softmax(Z)
$$

where:

```txt
P_hat: m x K
```

Each row is a probability distribution over classes.

Implementation convention in ML Core:

```txt
rows:
    samples

columns:
    classes
```

So:

```txt
probabilities(i, k):
    predicted probability that sample i belongs to class k
```

---

### Multiclass target representation

For softmax regression, targets can be represented in two useful ways.

#### Class-index targets

A class-index target stores the correct class as a number:

```txt
y = [0, 2, 1, 2]
```

This means:

```txt
sample 0 belongs to class 0
sample 1 belongs to class 2
sample 2 belongs to class 1
sample 3 belongs to class 2
```

This representation is compact and convenient for metrics.

#### One-hot targets

A one-hot target stores one column per class:

```txt
class 0: [1, 0, 0]
class 1: [0, 1, 0]
class 2: [0, 0, 1]
```

For a dataset with `m` samples and `K` classes:

```txt
Y: m x K one-hot target matrix
```

Each row contains exactly one `1` and the rest `0`.

ML Core should support the implementation internally using one-hot targets for vectorized loss and gradients, while allowing experiments and metrics to use class-index vectors when convenient.

---

### Multiclass prediction

After softmax, the predicted class is the class with the highest probability.

For one sample:

$$
\hat{y} = \arg\max_k \hat{p}_k
$$

Example:

```txt
softmax probabilities:
[0.10, 0.75, 0.15]

predicted class:
1
```

Unlike binary logistic regression, multiclass classification usually does not use a single threshold like `0.5`.

Instead, it uses the highest-probability class.

However, probability thresholds can still be useful in special cases, such as rejecting uncertain predictions.

For Phase 4, ML Core should implement standard argmax prediction.

---

### Categorical cross-entropy

Binary logistic regression uses binary cross-entropy.

Softmax regression uses categorical cross-entropy.

If the true class is represented as a one-hot vector:

$$
y =
\begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix}
$$

and the predicted probabilities are:

$$
\hat{p} =
\begin{bmatrix}
0.10 \\
0.75 \\
0.15
\end{bmatrix}
$$

then the loss is:

$$
L = -\sum_{k=1}^{K} y_k \log(\hat{p}_k)
$$

Because only the true class has value `1`, this becomes:

$$
L = -\log(\hat{p}_{true})
$$

For the full dataset:

$$
CE =
-\frac{1}{m}
\sum_{i=1}^{m}
\sum_{k=1}^{K}
y_{ik}\log(\hat{p}_{ik})
$$

This is the multiclass generalization of binary cross-entropy.

---

### Vectorized categorical cross-entropy

Using matrices:

```txt
Y: m x K one-hot target matrix
P: m x K predicted probability matrix
```

categorical cross-entropy can be written as:

$$
CE = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}Y_{ik}\log(P_{ik})
$$

In implementation terms:

```cpp
loss = -mean(rowwise_sum(Y * log(P)))
```

where multiplication and logarithm are element-wise operations.

As with binary cross-entropy, probabilities should be clipped:

```txt
epsilon <= P_ik <= 1 - epsilon
```

This avoids:

```txt
log(0)
```

and keeps the loss finite.

---

### Gradient for softmax regression

For softmax regression with categorical cross-entropy, the gradient has a clean form.

Let:

```txt
P: m x K predicted probability matrix
Y: m x K one-hot target matrix
```

Then:

$$
\nabla_W CE = \frac{1}{m}X^\top(P - Y)
$$

and:

$$
\nabla_b CE = \frac{1}{m}\sum_{i=1}^{m}(P_i - Y_i)
$$

where the bias gradient is computed column-wise across classes.

This mirrors binary logistic regression:

```txt
binary logistic regression:
    error = probabilities - targets

softmax regression:
    error = probability matrix - one-hot target matrix
```

This is why softmax + categorical cross-entropy is the standard multiclass pairing.

---

### Ridge regularization for softmax regression

Softmax regression can also use Ridge regularization.

The unregularized objective is:

$$
CE(W,b)
$$

With Ridge:

$$
J(W,b) = CE(W,b) + \lambda \|W\|_F^2
$$

where:

```txt
||W||_F²:
    sum of squared entries of the weight matrix
```

The Ridge gradient contribution is:

$$
2\lambda W
$$

So:

$$
\nabla_W J = \frac{1}{m}X^\top(P - Y) + 2\lambda W
$$

The bias is not regularized.

This matches the decision used for linear regression and binary logistic regression.

---

### Numerical stability

Softmax should be implemented carefully.

The direct formula:

$$
\frac{e^{z_k}}{\sum_j e^{z_j}}
$$

can overflow when logits are large.

A standard stable version subtracts the maximum logit before exponentiating:

$$
\hat{p}_k =
\frac{e^{z_k - z_{max}}}
{\sum_j e^{z_j - z_{max}}}
$$

where:

$$
z_{max} = \max_j z_j
$$

This does not change the softmax result, but it improves numerical stability.

For a matrix of logits, this maximum is computed row by row.

ML Core should implement vectorized softmax with row-wise max subtraction.

---

### Binary logistic regression as a special case

Binary logistic regression can be seen as a special case of multiclass logistic regression with two classes.

However, the implementation is usually simpler with sigmoid:

```txt
binary classification:
    one logit
    sigmoid
    binary cross-entropy

multiclass classification:
    K logits
    softmax
    categorical cross-entropy
```

ML Core implements both:

```txt
LogisticRegression:
    binary classifier using sigmoid

SoftmaxRegression:
    multiclass classifier using softmax
```

This keeps the binary and multiclass APIs explicit.

---

### Softmax and neural networks

Softmax is also important because it appears frequently in neural networks.

A typical neural network classifier ends with:

```txt
linear layer
softmax
cross-entropy loss
```

Conceptually:

```txt
hidden layers learn representations
final linear layer produces class logits
softmax converts logits into class probabilities
cross-entropy trains the classifier
```

So the path is:

```txt
linear regression:
    linear output for continuous targets

binary logistic regression:
    linear logit + sigmoid for binary classification

softmax regression:
    class logits + softmax for multiclass classification

neural network classifier:
    learned nonlinear representation + final logits + softmax
```

This makes softmax a key bridge from classical ML to Deep Learning.

---

### ML Core implementation decision

For Phase 4, ML Core should implement softmax regression, not only document it.

The implementation should include:

```txt
SoftmaxRegression
stable row-wise softmax
categorical cross-entropy
multiclass prediction with argmax
Ridge regularization support
multiclass metrics and evaluation support
multiclass experiments
```

The implementation should remain intentionally simple and consistent with the existing binary logistic regression design.

Rules:

```txt
weights:
    matrix with shape n_features x num_classes

bias:
    vector with shape num_classes

probabilities:
    matrix with shape num_samples x num_classes

predicted classes:
    vector of class indices

regularization:
    None or Ridge supported
    Lasso rejected for now

bias regularization:
    never regularize bias
```

This completes Phase 4 as a classification phase covering both binary and multiclass linear classifiers.
# Statistical Model, Estimators, and Literature

This document sets out the continuous-time observation model, derives the
pairwise covariance-rate estimators and their finite-sample moments, analyzes
the effects of microstructure noise and sparse observation graphs, and situates
these methods within the econometrics and statistical literature.

---

## 1. Continuous-Time Observation Model

### Latent process

Let $`X(t) \in \mathbb{R}^N`$ denote the latent vector of log prices for $`N`$
assets over continuous time $`t \in \mathbb{R}`$, governed by the
constant-coefficient Itô diffusion

```math
X(t) = X(0) + \mu t + L W(t), \qquad \Sigma = L L^\top \in \mathbb{R}^{N \times N},
```

where:

- $`\mu = (\mu_0, \dots, \mu_{N-1})^\top \in \mathbb{R}^N`$ is the vector of
  constant drift rates (log return per unit time),
- $`W(t)`$ is a standard $`N`$-dimensional Brownian motion, and
- $`\Sigma \succeq 0`$ is the symmetric positive-semidefinite **covariance rate**
  matrix (log-return covariance per unit time).

Over any time interval $`[s, u)`$ of duration $`u - s > 0`$, the latent increment
$`X(u) - X(s)`$ is Gaussian with mean $`\mu (u - s)`$ and covariance
$`\Sigma (u - s)`$. More generally, for two half-open intervals
$`I_e = [s_e, u_e)`$ and $`I_f = [s_f, u_f)`$ with overlap duration

```math
o_{ef} = \lvert I_e \cap I_f \rvert = \max(0, \min(u_e, u_f) - \max(s_e, s_f)),
```

the independent-increments property of Brownian motion implies

```math
\mathrm{Cov}\!\big(X_a(u_e) - X_a(s_e),\, X_b(u_f) - X_b(s_f)\big) = o_{ef}\,\Sigma_{ab}.
```

### Asynchronous noisy observations

Each asset $`a \in \{0, \dots, N-1\}`$ is observed at a sequence of strictly
increasing timestamps

```math
t_{a,0} < t_{a,1} < \dots < t_{a,m_a}.
```

The observation schedules $`\mathcal{T} = \{\{t_{a,k}\}_{k=0}^{m_a}\}_{a=0}^{N-1}`$
need not coincide across assets. At timestamp $`t_{a,k}`$, we observe the noisy
log price

```math
Y_a(t_{a,k}) = X_a(t_{a,k}) + \varepsilon_{a,k},
```

where the point measurement errors $`\varepsilon_{a,k}`$ satisfy the following
standing assumptions:

1. **Exogenous sampling**: The observation times $`\mathcal{T}`$ are independent
   of the Brownian motion $`W`$ and the measurement errors $`\varepsilon`$.
2. **Centered, independent point errors**: The errors $`\varepsilon_{a,k}`$ are
   independent of $`W`$, mutually independent across all assets $`a`$ and
   observation indices $`k`$, with
   $`\mathbb{E}[\varepsilon_{a,k} \mid \mathcal{T}] = 0`$ and asset-specific
   point-error variance
   $`\mathrm{Var}(\varepsilon_{a,k} \mid \mathcal{T}) = \tau_a^2 \ge 0`$.

### Increments and interval geometry

For each asset $`a`$, consecutive retained observations $`(t_{a,k-1}, t_{a,k})`$
($`k = 1, \dots, m_a`$) define a half-open increment interval

```math
I_e = [s_e, u_e) = [t_{a,k-1}, t_{a,k}), \qquad \ell_e = u_e - s_e > 0,
```

with observed return $`r_e = Y_a(u_e) - Y_a(s_e)`$ and drift-centered return

```math
\tilde{r}_e = r_e - \mu_a \ell_e = \big(X_a(u_e) - X_a(s_e) - \mu_a \ell_e\big) + \big(\varepsilon_{a,k} - \varepsilon_{a,k-1}\big).
```

Let $`\mathcal{I}_a`$ denote the set of active increments for asset $`a`$, and
let

```math
\mathcal{P}_{ab} = \big\{(e, f) \in \mathcal{I}_a \times \mathcal{I}_b : o_{ef} > 0\big\}
```

denote the set of overlapping increment pairs between assets $`a`$ and $`b`$.
Two geometric properties of the half-open intervals $`\mathcal{I}_a`$ simplify
the structure of $`\mathcal{P}_{ab}`$:

- **Single-asset disjointness ($`a = b`$)**: Because the timestamps of asset
  $`a`$ are strictly increasing, distinct intervals $`I_e, I_f \in \mathcal{I}_a`$
  ($`e \neq f`$) are disjoint ($`o_{ef} = 0`$). Adjacent intervals
  $`[t_{a,k-1}, t_{a,k})`$ and $`[t_{a,k}, t_{a,k+1})`$ touch at $`t_{a,k}`$ and
  have zero overlap. Thus $`\mathcal{P}_{aa} = \{(e, e) : e \in \mathcal{I}_a\}`$,
  with $`o_{ee} = \ell_e`$.
- **Cross-asset overlap partition ($`a \neq b`$)**: Because $`\mathcal{I}_a`$ is
  a family of pairwise disjoint intervals and $`\mathcal{I}_b`$ is a family of
  pairwise disjoint intervals, their nonempty intersections
  $`\{I_e \cap I_f\}_{(e,f) \in \mathcal{P}_{ab}}`$ are pairwise disjoint and
  partition the joint observation support:

```math
\bigsqcup_{(e,f) \in \mathcal{P}_{ab}} (I_e \cap I_f)
= \left(\bigsqcup_{e \in \mathcal{I}_a} I_e\right) \cap \left(\bigsqcup_{f \in \mathcal{I}_b} I_f\right).
```

In particular, the sum of unweighted overlaps
$`\sum_{(e,f) \in \mathcal{P}_{ab}} o_{ef}`$ is the Lebesgue measure of the time
set over which both assets are simultaneously covered by active increments.

---

## 2. Pairwise Covariance-Rate Estimators

### Conditional moment equation

For any two increments $`e \in \mathcal{I}_a`$ and $`f \in \mathcal{I}_b`$, let
$`\Gamma_{ef}`$ denote the conditional covariance of their return measurement
errors given $`\mathcal{T}`$. Under independent point errors:

- When $`a = b`$ and $`e = f`$, the return error
  $`\varepsilon_{a,k} - \varepsilon_{a,k-1}`$ is the difference of two
  independent errors of variance $`\tau_a^2`$, so $`\Gamma_{ee} = 2\tau_a^2`$.
- When $`a = b`$ and $`e, f`$ are adjacent increments sharing a single endpoint
  $`t_{a,k}`$, the shared error appears with opposite signs in the two returns,
  giving $`\Gamma_{ef} = -\tau_a^2`$ (while $`o_{ef} = 0`$).
- When $`a \neq b`$, point errors are independent across assets, so
  $`\Gamma_{ef} = 0`$.

Consequently, for every overlapping pair $`(e, f) \in \mathcal{P}_{ab}`$, the
noise-adjusted centered product

```math
Z_{ef} = (r_e - \mu_a \ell_e)(r_f - \mu_b \ell_f) - \Gamma_{ef}, \qquad
\Gamma_{ef} = \begin{cases}
2\tau_a^2 & \text{if } a = b \text{ (so } e = f\text{)}, \\
0 & \text{if } a \neq b,
\end{cases}
```

satisfies the conditional moment identity

```math
\mathbb{E}[Z_{ef} \mid \mathcal{T}] = o_{ef}\,\Sigma_{ab}.
```

### General weighted linear overlap estimators

Given any nonnegative timestamp-measurable weights $`w_{ef}`$ and pair-specific
factors $`h_{ef} > 0`$, any linear estimator of the form

```math
\widehat{\Sigma}_{ab}(h)
= \frac{\sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef} h_{ef} o_{ef} Z_{ef}}
       {\sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef} h_{ef} o_{ef}^2}
```

is conditionally unbiased for $`\Sigma_{ab}`$ given $`\mathcal{T}`$ whenever the
denominator is positive ($`\mathcal{P}_{ab} \neq \emptyset`$):

```math
\mathbb{E}\!\left[\widehat{\Sigma}_{ab}(h) \,\middle|\, \mathcal{T}\right]
= \frac{\sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef} h_{ef} o_{ef} (o_{ef} \Sigma_{ab})}
       {\sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef} h_{ef} o_{ef}^2}
= \Sigma_{ab}.
```

The package implements two canonical choices of $`h_{ef}`$.

### Hayashi–Yoshida covariance-rate estimator (`method="hy"`)

Setting $`h_{ef} = 1 / o_{ef}`$ yields the Hayashi–Yoshida covariance-rate
estimator

```math
\widehat{\Sigma}^{\mathrm{hy}}_{ab}
= \frac{\sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef} Z_{ef}}
       {\sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef} o_{ef}}.
```

When $`w_{ef} \equiv 1`$ and $`\mu = 0, \tau = 0`$, the numerator
$`\sum_{(e,f) \in \mathcal{P}_{ab}} r_e r_f`$ is the integrated covariance
estimator of Hayashi and Yoshida (2005), and the denominator
$`\sum_{(e,f) \in \mathcal{P}_{ab}} o_{ef}`$ is the total joint observation
coverage. Dividing by the active overlap exposure $`\sum w_{ef} o_{ef}`$ rather
than a nominal clock horizon $`T`$ ensures that
$`\widehat{\Sigma}^{\mathrm{hy}}_{ab}`$ remains conditionally unbiased for the
rate $`\Sigma_{ab}`$ under arbitrary arrival gaps, partial coverage, and
exponential decay $`w_{ef}`$.

On the diagonal ($`a = b`$), $`\mathcal{P}_{aa} = \{(e, e)\}`$ and
$`o_{ee} = \ell_e`$, so

```math
\widehat{\Sigma}^{\mathrm{hy}}_{aa}
= \frac{\sum_{e \in \mathcal{I}_a} w_{ee} (\tilde{r}_e^2 - 2\tau_a^2)}
       {\sum_{e \in \mathcal{I}_a} w_{ee} \ell_e}.
```

### Duration-normalized overlap estimator (`method="overlap"`)

Setting $`h_{ef} = 1 / (\ell_e \ell_f)`$ yields the duration-normalized overlap
estimator

```math
\widehat{\Sigma}^{\mathrm{overlap}}_{ab}
= \frac{\sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef} \frac{o_{ef}}{\ell_e \ell_f} Z_{ef}}
       {\sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef} \frac{o_{ef}^2}{\ell_e \ell_f}}.
```

To see the statistical motivation for $`h_{ef} = (\ell_e \ell_f)^{-1}`$, define
the duration-standardized returns and the dimensionless overlap fraction

```math
\xi_e = \frac{\tilde{r}_e}{\sqrt{\ell_e}}, \qquad
\xi_f = \frac{\tilde{r}_f}{\sqrt{\ell_f}}, \qquad
\rho_{ef} = \frac{o_{ef}}{\sqrt{\ell_e \ell_f}} \in (0, 1].
```

In the noiseless model ($`\tau = 0`$), $`\xi_e \sim \mathcal{N}(0, \Sigma_{aa})`$
and $`\xi_f \sim \mathcal{N}(0, \Sigma_{bb})`$ have constant variances across
all increments regardless of their durations $`\ell_e, \ell_f`$, and their
product satisfies

```math
\mathbb{E}[\xi_e \xi_f \mid \mathcal{T}] = \Sigma_{ab} \rho_{ef}, \qquad
\mathrm{Var}(\xi_e \xi_f \mid \mathcal{T}) = \Sigma_{aa} \Sigma_{bb} + \Sigma_{ab}^2 \rho_{ef}^2.
```

In terms of $`\xi_e, \xi_f`$ and $`\rho_{ef}`$, the estimator is the weighted
least-squares regression slope through the origin of the noise-adjusted
standardized product on $`\rho_{ef}`$:

```math
\widehat{\Sigma}^{\mathrm{overlap}}_{ab}
= \frac{\sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef} \rho_{ef} \left(\xi_e \xi_f - \frac{\Gamma_{ef}}{\sqrt{\ell_e \ell_f}}\right)}
       {\sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef} \rho_{ef}^2}.
```

Two special cases illustrate its behavior:

- **Variance optimality at zero correlation**: For a family of independent
  noiseless increment pairs with $`w_{ef} = 1`$, when $`\Sigma_{ab} = 0`$, the
  raw products $`Z_{ef} = \tilde{r}_e \tilde{r}_f`$ have conditional variance
  $`\mathrm{Var}(Z_{ef} \mid \mathcal{T}) = \Sigma_{aa}\Sigma_{bb} \ell_e \ell_f`$.
  By the Gauss–Markov theorem, the inverse-variance weight
  $`h_{ef} = (\ell_e \ell_f)^{-1}`$ minimizes the conditional variance of
  $`\widehat{\Sigma}_{ab}(h)`$ within this linear class.
- **Diagonal efficiency under irregular spacing**: On the diagonal
  ($`a = b, e = f`$), $`\rho_{ee} = 1`$, so

```math
\widehat{\Sigma}^{\mathrm{overlap}}_{aa}
= \frac{\sum_{e \in \mathcal{I}_a} w_{ee} \frac{\tilde{r}_e^2 - 2\tau_a^2}{\ell_e}}
       {\sum_{e \in \mathcal{I}_a} w_{ee}}.
```

When $`\tau_a = 0`$ and $`w_{ee} = 1`$, the standardized squared returns
$`\tilde{r}_e^2 / \ell_e \sim \Sigma_{aa} \chi^2_1`$ are independent and
identically distributed with variance $`2\Sigma_{aa}^2`$ across increments of
arbitrary durations $`\ell_e`$. Their simple average
$`\widehat{\Sigma}^{\mathrm{overlap}}_{aa}`$ is the minimum-variance unbiased
(and Gaussian maximum-likelihood) estimator of $`\Sigma_{aa}`$, whereas
$`\widehat{\Sigma}^{\mathrm{hy}}_{aa}`$ weights each $`\tilde{r}_e^2 / \ell_e`$
by $`\ell_e`$ and is less efficient when interval lengths are imbalanced.

When observation times are synchronous and equally spaced
($`\ell_e = \ell_f = o_{ef} = \Delta`$), the two estimators coincide identically:
$`\widehat{\Sigma}^{\mathrm{hy}}_{ab} = \widehat{\Sigma}^{\mathrm{overlap}}_{ab}`$.

---

## 3. Finite-Sample Variance and Temporal Weighting

### Exact conditional variance under Gaussian noise

Let $`x \in \mathbb{R}^{m_a}`$ and $`y \in \mathbb{R}^{m_b}`$ be the vectors of
centered returns $`\tilde{r}_e`$ ($`e \in \mathcal{I}_a`$) and $`\tilde{r}_f`$
($`f \in \mathcal{I}_b`$). When the point measurement errors
$`\varepsilon_{a,k}`$ are also Gaussian (and independent of $`W`$ and
$`\mathcal{T}`$), the pair $`(x, y)`$ is jointly Gaussian conditional on
$`\mathcal{T}`$ with zero mean and block covariance matrices

```math
A = \mathrm{Cov}(x \mid \mathcal{T}), \qquad
B = \mathrm{Cov}(y \mid \mathcal{T}), \qquad
C = \mathrm{Cov}(x, y \mid \mathcal{T}).
```

Here $`A_{ee'} = \Sigma_{aa} \ell_e \mathbf{1}_{\{e=e'\}} + \Gamma_{ee'}`$ is
tridiagonal with diagonal $`\Sigma_{aa}\ell_e + 2\tau_a^2`$ and first
off-diagonal $`-\tau_a^2`$, and
$`B_{ff'} = \Sigma_{bb} \ell_f \mathbf{1}_{\{f=f'\}} + \Gamma_{ff'}`$ is
tridiagonal with diagonal $`\Sigma_{bb}\ell_f + 2\tau_b^2`$ and first
off-diagonal $`-\tau_b^2`$. For distinct assets ($`a \neq b`$) under independent
point errors, $`C_{ef} = \Sigma_{ab} o_{ef}`$; for a single asset ($`a = b`$),
$`x = y`$ and $`C = A = B`$ includes the endpoint-noise covariance
$`\Gamma_{ee'}`$.

Any linear overlap estimator has a numerator of the form
$`S = x^\top W y - \text{const}`$ for a sparse weight matrix
$`W \in \mathbb{R}^{m_a \times m_b}`$ supported on $`\mathcal{P}_{ab}`$. By
Isserlis' (Wick's) theorem for centered jointly Gaussian vectors,

```math
\mathrm{Var}(S \mid \mathcal{T})
= \mathrm{tr}(W^\top A W B) + \mathrm{tr}\!\big((W^\top C)^2\big).
```

Dividing $`\mathrm{Var}(S \mid \mathcal{T})`$ by the squared estimator
denominator yields the exact conditional variance of $`\widehat{\Sigma}_{ab}`$.

This identity shows why neither the overlap pair count
$`\lvert \mathcal{P}_{ab} \rvert`$ (`counts`) nor the total overlap duration
$`\sum o_{ef}`$ (`coverage`) equals an effective sample size:

- **Shared intervals**: If a single long increment $`I_e`$ of a slowly traded
  asset $`a`$ overlaps $`k`$ short increments $`I_{f_1}, \dots, I_{f_k}`$ of a
  frequently traded asset $`b`$, all $`k`$ products
  $`\tilde{r}_e \tilde{r}_{f_j}`$ reuse the same random return $`\tilde{r}_e`$.
  In $`\mathrm{tr}(W^\top A W B)`$, these $`k`$ pairs couple through the
  diagonal entry $`A_{ee} = \Sigma_{aa} \ell_e + 2\tau_a^2`$, so precision is
  limited by the single observation of asset $`a`$ over $`I_e`$.
- **Adjacent-endpoint noise correlation**: When $`\tau_a > 0`$, the off-diagonal
  entries $`A_{e,e+1} = -\tau_a^2`$ couple consecutive increments in
  $`\mathrm{tr}(W^\top A W B)`$.

### Exponential decay and time-varying covariance rates

When a half-life $`H > 0`$ is specified, the weight assigned at watermark
timestamp $`t`$ to an overlapping pair $`(e, f) \in \mathcal{P}_{ab}`$ is
anchored at the midpoint $`m_{ef} = \max(s_e, s_f) + \frac{1}{2} o_{ef}`$ of the
overlap interval $`I_e \cap I_f`$:

```math
w_{ef}(t) = 2^{-(t - m_{ef}) / H} = \exp\!\left(-\frac{\ln 2}{H}(t - m_{ef})\right).
```

Because $`m_{ef}`$ depends only on the interval endpoints, advancing the
watermark from $`t`$ to $`t' \ge t`$ scales every historical weight by the
common factor $`\exp(-\frac{\ln 2}{H}(t' - t))`$.

If the latent covariance rate $`\Sigma(s)`$ is a deterministic integrable
function of time $`s`$ (or if we condition on both $`\mathcal{T}`$ and a
stochastic covariance path $`\Sigma(\cdot)`$ that is independent of the driving
Brownian motion $`W`$ and point errors $`\varepsilon`$), the conditional
expectation of a return product is the integrated covariance over the
intersection:

```math
\mathbb{E}[Z_{ef} \mid \mathcal{T}, \Sigma(\cdot)] = \int_{I_e \cap I_f} \Sigma_{ab}(s)\,ds.
```

Consequently, under time-varying volatility,
$`\widehat{\Sigma}^{\mathrm{hy}}_{ab}`$ and
$`\widehat{\Sigma}^{\mathrm{overlap}}_{ab}`$ estimate weighted averages of
$`\Sigma_{ab}(s)`$ against two different pair-dependent temporal kernels:

```math
K^{\mathrm{hy}}_{ab}(s) \propto \sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef}(t)\,\mathbf{1}_{I_e \cap I_f}(s), \qquad
K^{\mathrm{overlap}}_{ab}(s) \propto \sum_{(e,f) \in \mathcal{P}_{ab}} w_{ef}(t)\,\frac{o_{ef}}{\ell_e \ell_f}\,\mathbf{1}_{I_e \cap I_f}(s).
```

Neither estimator recovers an instantaneous spot covariance $`\Sigma(t)`$
without a local stationarity assumption over the effective support of the
kernel.

### Whole-increment window expiration

When a sliding window of width $`W > 0`$ is active at watermark $`t`$, the
engine retains all increments $`I_e = [s_e, u_e)`$ whose start time satisfies
$`s_e \ge t - W`$ and expires any increment with $`s_e < t - W`$ in its
entirety.

Expiring whole increments—rather than retaining an increment that straddles the
cutoff $`c = t - W`$ and clipping its overlap duration to
$`\tilde{o}_{ef} = \lvert I_e \cap I_f \cap [c, t) \rvert`$—is necessary to
preserve conditional unbiasedness. If $`I_e`$ and $`I_f`$ overlap both before
and after $`c`$, the observed return product $`Z_{ef}`$ still accumulates latent
Brownian covariation over the full intersection $`I_e \cap I_f`$ of length
$`o_{ef}`$. Dividing $`Z_{ef}`$ by a truncated denominator
$`\tilde{o}_{ef} < o_{ef}`$ would yield conditional expectation
$`(o_{ef} / \tilde{o}_{ef})\Sigma_{ab}`$, scaling the true covariance rate by
the factor $`o_{ef} / \tilde{o}_{ef} > 1`$ whenever $`\Sigma_{ab} \neq 0`$.

---

## 4. Microstructure Noise and Sampling Frequency

### Diagonal bias correction and first-order autocovariance

Under additive point noise with variance $`\tau_a^2`$, each squared return of
asset $`a`$ has conditional expectation

```math
\mathbb{E}[\tilde{r}_e^2 \mid \mathcal{T}] = \Sigma_{aa}\ell_e + 2\tau_a^2.
```

If $`n`$ increments are observed over a fixed horizon $`T`$, the uncorrected sum
of squared returns $`\sum_{e=1}^n \tilde{r}_e^2`$ has expectation
$`\Sigma_{aa} T + 2 n \tau_a^2`$, which grows linearly with $`n`$ as sampling
becomes finer. Subtracting $`2\tau_a^2`$ per increment removes this bias
exactly when $`\tau_a^2`$ is known.

To estimate $`\tau_a^2`$ from the observations of asset $`a`$, consider two
consecutive increments $`I_k = [t_{a,k-1}, t_{a,k})`$ and
$`I_{k+1} = [t_{a,k}, t_{a,k+1})`$. Under exogenous sampling, independent
latent increments, correct drift centering at $`\mu_a`$, and mutually
independent point errors, the latent increments over the disjoint intervals
$`I_k`$ and $`I_{k+1}`$ are independent and zero-mean, while their measurement
errors $`\varepsilon_{a,k} - \varepsilon_{a,k-1}`$ and
$`\varepsilon_{a,k+1} - \varepsilon_{a,k}`$ share only the common middle
endpoint error $`\varepsilon_{a,k}`$ with opposite signs. Therefore,

```math
\mathbb{E}[\tilde{r}_{a,k} \tilde{r}_{a,k+1} \mid \mathcal{T}]
= \mathbb{E}\!\big[(\varepsilon_{a,k} - \varepsilon_{a,k-1})(\varepsilon_{a,k+1} - \varepsilon_{a,k}) \,\big|\, \mathcal{T}\big]
= -\tau_a^2.
```

The function `estimate_noise` evaluates the corresponding sample moment

```math
\widehat{\tau}_a^2 = -\frac{1}{m_a - 1} \sum_{k=1}^{m_a - 1} \tilde{r}_{a,k} \tilde{r}_{a,k+1}.
```

In finite samples where latent volatility dominates microstructure noise,
$`\widehat{\tau}_a^2`$ can be negative; `estimate_noise` returns the signed
moment without clipping so that the caller can inspect whether noise is
detectable relative to sampling error.

### Variance divergence under noisy infill sampling

Subtracting $`2\tau_a^2`$ removes the diagonal bias, and independent point noise
across assets ($`a \neq b`$) introduces zero bias into off-diagonal products
($`\Gamma_{ef} = 0`$). Even so, raw return products are **not** consistent under
noisy infill asymptotics ($`n \to \infty`$ on a fixed horizon $`T`$ with fixed
$`\tau_a^2, \tau_b^2 > 0`$).

To see this explicitly, consider two assets $`a \neq b`$ observed synchronously
on the uniform grid $`t_k = k T / n`$ ($`k = 0, \dots, n`$) with spacing
$`\Delta = T / n`$ and Gaussian point errors. Both `"hy"` and `"overlap"` reduce
to $`\widehat{\Sigma}_{ab} = T^{-1} \sum_{k=1}^n \tilde{r}_{a,k} \tilde{r}_{b,k}`$.
Here $`A = \mathrm{Cov}(\tilde{r}_a)`$ is the $`n \times n`$ Toeplitz
tridiagonal matrix with diagonal $`\Sigma_{aa}\Delta + 2\tau_a^2`$ and first
off-diagonal $`-\tau_a^2`$, $`B = \mathrm{Cov}(\tilde{r}_b)`$ is analogous, and
$`C = \Sigma_{ab} \Delta I_n`$. Applying the Gaussian variance formula with
$`W = T^{-1} I_n`$ gives the exact finite-$`n`$ variance

```math
\mathrm{Var}(\widehat{\Sigma}_{ab})
= \frac{1}{T^2} \Big(\mathrm{tr}(A B) + \mathrm{tr}(C^2)\Big)
= \frac{1}{T^2} \left[n (\Sigma_{aa}\Delta + 2\tau_a^2)(\Sigma_{bb}\Delta + 2\tau_b^2) + 2(n - 1)\tau_a^2 \tau_b^2 + n \Sigma_{ab}^2 \Delta^2\right].
```

Substituting $`\Delta = T / n`$ and collecting powers of $`n`$ yields

```math
\mathrm{Var}(\widehat{\Sigma}_{ab})
= \frac{2(3n - 1)\tau_a^2 \tau_b^2}{T^2} + \frac{2(\Sigma_{aa}\tau_b^2 + \Sigma_{bb}\tau_a^2)}{T} + \frac{\Sigma_{aa}\Sigma_{bb} + \Sigma_{ab}^2}{n}.
```

As $`n \to \infty`$ with $`T`$ and $`\tau_a^2, \tau_b^2 > 0`$ fixed, the leading
term is $`6 n \tau_a^2 \tau_b^2 / T^2`$, so the root-mean-square error of raw
noisy products grows as $`\sqrt{6 n}\,\tau_a \tau_b / T`$.

### Fixed-bin thinning (`coarsen`) versus pre-averaging

One elementary way to bound the $`O(n)`$ noise variance growth in batch data is
to thin the observation frequency before forming increments. Given a bin width
$`\Delta > 0`$ and origin $`t_0`$, `coarsen` partitions the time axis into
half-open buckets $`[t_0 + m\Delta, t_0 + (m+1)\Delta)`$ ($`m \in \mathbb{Z}`$)
and retains at most one observation per asset in each bucket—specifically, the
observation with the largest timestamp (and latest arrival index on ties),
preserving its exact original timestamp.

Over a finite time span of length $`T`$, observations can intersect at most
$`\lfloor T / \Delta \rfloor + 2`$ buckets, bounding each asset to at most
$`\lfloor T / \Delta \rfloor + 2`$ retained points and at most
$`\lfloor T / \Delta \rfloor + 1`$ increments, trading latent-return sample size
against the noise variance term. Two properties distinguish fixed-bin thinning
from pre-averaging:

- Because original timestamps are preserved, an observation near the end of
  bucket $`m - 1`$ and an observation near the start of bucket $`m`$ can still
  be separated by less than $`\Delta`$.
- Thinning discards intermediate observations rather than averaging them.
  **Pre-averaging** (Jacod et al., 2009; Christensen, Kinnebrock, and Podolskij,
  2010; Christensen, Podolskij, and Vetter, 2013) forms local linear
  combinations of returns $`R_e = \sum_j a_j r_j`$ over blocks of length
  $`k_n \propto n^{1/2}`$, which alters both the latent covariance exposure to
  $`\int g_e(s) g_f(s)\,ds`$ (where $`g_e(s) = \sum_j a_j \mathbf{1}_{I_j}(s)`$)
  and the endpoint noise quadratic form, achieving the $`n^{-1/4}`$ convergence
  rate under noisy infill asymptotics.

---

## 5. Low-Rank-Plus-Diagonal Factor Completion and Identifiability

### Constrained weighted graph objective

A `MomentSnapshot` provides diagonal variance-rate estimates
$`v = (v_0, \dots, v_{N-1})^\top`$ and off-diagonal covariance-rate estimates
$`\widehat{\Sigma}_{ij}`$ on the subset of unordered pairs
$`\mathcal{E} \subseteq \{\{i, j\} : 0 \le i < j < N\}`$ that have a positive
overlap denominator and positive fitting weight $`w_{ij} > 0`$. Because pairs
are estimated from distinct sets of overlapping intervals (or may have no
overlap at all), the raw matrix need not be positive semidefinite and may have
missing entries.

To obtain a positive-definite covariance operator across all $`N`$ assets,
`fit_factor` fits a rank-$`K`$ factor-plus-diagonal model

```math
\Sigma_{\mathrm{model}} = B B^\top + \mathrm{diag}(d), \qquad
B \in \mathbb{R}^{N \times K}, \quad d \in \mathbb{R}_{>0}^N,
```

subject to exact preservation of the supplied marginal variances $`v_i > 0`$ and
a relative floor $`\eta \in (0, 1]`$ (`specific_floor`) on each asset's specific
variance:

```math
(\Sigma_{\mathrm{model}})_{ii} = \|B_{i,:}\|_2^2 + d_i = v_i, \qquad d_i \ge \eta v_i > 0 \quad (i = 0, \dots, N-1).
```

Introducing standardized factor rows
$`z_i = B_{i,:}^\top / \sqrt{v_i} \in \mathbb{R}^K`$ and standardized target
correlations $`c_{ij} = \widehat{\Sigma}_{ij} / \sqrt{v_i v_j}`$, the constraint
$`d_i = v_i(1 - \|z_i\|_2^2) \ge \eta v_i`$ is equivalent to the Euclidean ball
constraint $`\|z_i\|_2 \le R = \sqrt{1 - \eta}`$. The solver minimizes the
weighted squared correlation error over observed edges:

```math
\min_{z_0, \dots, z_{N-1} \in \mathbb{R}^K} \;
f(Z) = \frac{1}{\sum_{\{i,j\} \in \mathcal{E}} w_{ij}} \sum_{\{i,j\} \in \mathcal{E}} w_{ij} \left(z_i^\top z_j - c_{ij}\right)^2
\quad \text{subject to} \quad \|z_i\|_2 \le R \;\; (i = 0, \dots, N-1).
```

Unobserved pairs $`\{i, j\} \notin \mathcal{E}`$ do not enter the sum (they
carry weight zero, which is distinct from an observed target $`c_{ij} = 0`$).
Regardless of where the optimization terminates, the reconstructed covariance
matrix satisfies

```math
\Sigma_{\mathrm{model}} = B B^\top + \mathrm{diag}(d) \succeq \mathrm{diag}(d) \succeq \eta\,\mathrm{diag}(v) \succ 0.
```

### Structural obstructions on sparse graphs

When $`\mathcal{E}`$ is a proper subset of all $`\binom{N}{2}`$ pairs,
minimizing $`f(Z)`$ on $`\mathcal{E}`$ does not guarantee that the implied
off-diagonal covariances $`B_{i,:} B_{j,:}^\top = \sqrt{v_i v_j}\, z_i^\top z_j`$
on unobserved pairs $`\{i, j\} \notin \mathcal{E}`$ are uniquely identified.
`FactorFit.diagnostics` inspects the observation graph
$`\mathcal{G} = (\{0, \dots, N-1\}, \mathcal{E})`$ and reports three structural
obstructions when $`K \ge 1`$:

- **Disconnected components (`n_graph_components > 1`)**: Suppose $`\mathcal{G}`$
  decomposes into connected components $`\mathcal{V}_1, \dots, \mathcal{V}_C`$
  with $`C > 1`$. For any independent orthogonal matrices
  $`Q_1, \dots, Q_C \in O(K)`$, the transformation $`z_i \mapsto Q_c z_i`$ for
  $`i \in \mathcal{V}_c`$ preserves every row norm $`\|z_i\|_2`$ and every
  intra-component inner product $`z_i^\top z_j`$ ($`\{i, j\} \in \mathcal{E}`$),
  leaving $`f(Z)`$ invariant. However, for $`i \in \mathcal{V}_c`$ and
  $`j \in \mathcal{V}_{c'}`$ ($`c \neq c'`$), the cross-component inner product
  $`z_i^\top Q_c^\top Q_{c'} z_j`$ depends on the arbitrary relative rotation
  $`Q_c^\top Q_{c'} \in O(K)`$.
- **Bipartite components (`n_bipartite_components > 0`)**: Suppose a connected
  component $`\mathcal{V}_c = \mathcal{A} \sqcup \mathcal{B}`$ is bipartite, so
  every observed edge $`\{i, j\} \in \mathcal{E}`$ in $`\mathcal{V}_c`$ has one
  endpoint in $`\mathcal{A}`$ and one in $`\mathcal{B}`$. Then for any scalar
  $`\alpha > 0`$ (or invertible matrix $`S \in \mathrm{GL}(K)`$) that keeps
  $`\|\alpha z_i\|_2 \le R`$ ($`i \in \mathcal{A}`$) and
  $`\|\alpha^{-1} z_j\|_2 \le R`$ ($`j \in \mathcal{B}`$), replacing
  $`z_i \mapsto \alpha z_i`$ on $`\mathcal{A}`$ and
  $`z_j \mapsto \alpha^{-1} z_j`$ on $`\mathcal{B}`$ leaves all observed
  bipartite products $`(\alpha z_i)^\top (\alpha^{-1} z_j) = z_i^\top z_j`$
  unchanged while scaling unobserved intra-partition products
  $`z_i^\top z_{i'}`$ ($`i, i' \in \mathcal{A}`$) by $`\alpha^2`$ and
  $`z_j^\top z_{j'}`$ ($`j, j' \in \mathcal{B}`$) by $`\alpha^{-2}`$.
- **Underdetermined row updates (`n_assets_with_degree_below_rank`)**: When
  updating row $`i`$ with neighbor set $`\mathcal{N}(i)`$, the row Hessian is
  $`G_i = \sum_{j \in \mathcal{N}(i)} \tilde{w}_{ij} z_j z_j^\top \in \mathbb{R}^{K \times K}`$,
  which has rank at most $`\min(K, \lvert \mathcal{N}(i) \rvert)`$. If
  $`0 < \lvert \mathcal{N}(i) \rvert < K`$, $`G_i`$ has a nullspace of dimension
  at least $`K - \lvert \mathcal{N}(i) \rvert \ge 1`$, so unconstrained interior
  solutions are not unique. (Isolated assets with
  $`\lvert \mathcal{N}(i) \rvert = 0`$ are fixed at $`z_i = 0`$.)

These three checks are structural graph warnings rather than universal
necessary or sufficient conditions for identification (for example, the
degenerate boundary case $`\eta = 1`$ forces $`R = 0`$ and uniquely yields
$`\Sigma_{\mathrm{model}} = \mathrm{diag}(v)`$ on any graph, while for
$`K \ge 1`$, `completion_identifiability` is always reported as
`"not_established"`). Moreover, because $`f(Z)`$ is nonconvex, alternating
minimization finds a local candidate rather than a certified global minimum.

---

## 6. Comparison of Asynchronous Covariance Estimators and Literature

### Summary of estimation frameworks

The table below compares the two estimators implemented in `async-covariance`
with alternative high-frequency covariance frameworks in the literature.

| Estimator / Framework | Statistical Target and Regime | Online State and Update Work | Structural Assumptions and Requirements |
| :--- | :--- | :--- | :--- |
| **Hayashi–Yoshida rate** (`method="hy"`) | Constant covariance rate $`\Sigma`$ from asynchronous returns with low microstructure noise | Interval index and 14 sufficient statistics per active pair; exact local insert/replace/delete | Exogenous sampling; variance of raw products grows as $`O(n)`$ under noisy infill sampling |
| **Overlap regression** (`method="overlap"`) | Constant covariance rate $`\Sigma`$, particularly under heterogeneous or imbalanced interval durations | Same interval index and 14 sufficient statistics as `"hy"` | Exogenous sampling; inverse-duration weights are variance-optimal at $`\Sigma_{ab}=0`$ without noise |
| **Pre-averaged HY / Realized kernels** | Integrated covariance $`\int_0^T \Sigma(s)\,ds`$ under dense, noisy, asynchronous point observations | Rolling observation buffer and multi-lag return sums | Requires bandwidth / kernel selection ($`k_n \propto n^{1/2}`$ for pre-averaging; $`H \propto n^{3/5}`$ for positive-semidefinite non-flat-top realized kernels), boundary handling, and refresh-time or pair-interpolation rules |
| **Spectral local method of moments (LMM)** | Integrated or smoothed spot covariance under dense noisy observations; asymptotically efficient | Block-wise local sine-basis coefficients and noise estimators | Common block size is constrained by the least liquid asset; optimal multivariate weights act on $`N^2 \times N^2`$ Kronecker forms |
| **Positive-semidefinite Fourier spot estimator** | Spot covariance $`\Sigma(t)`$ on asynchronous grids with fine mesh | Fixed-frequency trigonometric return coefficients per asset | Requires choosing frequency cutoffs, localization kernel, and horizon scaling; causal real-time evaluation faces boundary bias |
| **Gaussian QML / Kalman–EM** | Integrated covariance $`\int_0^T \Sigma(s)\,ds`$ (via a constant-parameter working Gaussian quasi-likelihood) and noise variance $`\tau^2`$ | Pairwise or joint Kalman filter and smoother state | Parameter updates and historical price revisions require full or windowed smoothing passes rather than $`O(1)`$ sufficient-statistic subtraction |
| **Local factor PCA / POET** | High-dimensional spot or integrated factor covariance on dense panels | Block-wise pre-averaging, eigendecomposition, and residual thresholding | Requires uniform observation density across the panel, pervasive factors, and sparse idiosyncratic covariance |

### Annotated primary references

1. **Hayashi, T., and N. Yoshida (2005)**, *On covariance estimation of
   non-synchronously observed diffusion processes*, Bernoulli 11(2), 359–379.
   ([Author PDF](https://www.ms.u-tokyo.ac.jp/~nakahiro/mypapers_for_personal_use/hayyos03.pdf))
   Introduces the overlap-indicator cross-product sum
   $`\sum_{o_{ef} > 0} r_e r_f`$ as a consistent, synchronization-free estimator
   of integrated covariance $`\int_0^T \Sigma_{ab}(s)\,ds`$ for noiseless
   diffusions under exogenous sampling. `method="hy"` normalizes this sum (and
   its exponentially decayed extension) by the active overlap exposure
   $`\sum w_{ef} o_{ef}`$ to estimate the covariance rate $`\Sigma_{ab}`$.

2. **Christensen, K., S. Kinnebrock, and M. Podolskij (2010)**, *Pre-averaging
   estimators of the ex-post covariance matrix in noisy diffusion models with
   non-synchronous data*, Journal of Econometrics 159(1), 116–133
   ([DOI](https://doi.org/10.1016/j.jeconom.2010.05.001),
   [arXiv:2602.19645](https://arxiv.org/abs/2602.19645)); and **Christensen,
   K., M. Podolskij, and M. Vetter (2013)**, *On covariation estimation for
   multivariate continuous Itô semimartingales with noise in non-synchronous
   observation schemes*, Journal of Multivariate Analysis 120, 59–84
   ([DOI](https://doi.org/10.1016/j.jmva.2013.05.002),
   [Preprint](https://eldorado.tu-dortmund.de/bitstreams/ad4155d5-17c4-4666-a8f6-e2f4edc3746b/download)).
   Establish how local pre-averaging of noisy asynchronous observations
   suppresses the $`O(n)`$ microstructure-noise variance divergence of raw HY
   products and achieves $`n^{-1/4}`$-consistent covariation estimation.

3. **Barndorff-Nielsen, O. E., P. R. Hansen, A. Lunde, and N. Shephard (2011)**,
   *Multivariate realised kernels: Consistent positive semi-definite estimators
   of the covariation of equity prices with noise and non-synchronous trading*,
   Journal of Econometrics 162(2), 149–169.
   ([DOI](https://doi.org/10.1016/j.jeconom.2010.07.009),
   [Preprint](https://pure.au.dk/ws/files/14416443/rp08_63.pdf))
   Synchronizes observations via all-asset refresh times and applies a common
   non-flat-top positive-semidefinite lag kernel (such as a Parzen kernel) with
   bandwidth $`H \propto n^{3/5}`$ to guarantee positive semidefiniteness and
   noise robustness at rate $`n^{-1/5}`$ (flat-top kernels achieve $`n^{-1/4}`$
   but can be indefinite). Using a common refresh schedule across all $`N`$
   assets discards observations down to the pace of the slowest asset, whereas
   pairwise refresh schedules forfeit the shared-grid positive-semidefiniteness
   guarantee.

4. **Holý, V., and P. Tomanová (2021)**, *Streaming Approach to Quadratic
   Covariation Estimation Using Financial Ultra-High-Frequency Data*,
   arXiv:2003.13062v3.
   ([arXiv](https://arxiv.org/abs/2003.13062v3))
   Formulates fixed-bandwidth streaming recurrences (Section 3.2) for realized
   kernels and pre-averaged estimators in $`O(N^2 + NH)`$ memory by fixing the
   lag horizon $`H`$ and omitting finite-sample boundary adjustments.

5. **Shephard, N., and D. Xiu (2017)**, *Econometric analysis of multivariate
   realised QML: Estimation of the covariation of equity prices under
   asynchronous trading*, Journal of Econometrics 201(1), 19–42.
   ([DOI](https://doi.org/10.1016/j.jeconom.2017.04.003),
   [Author PDF](https://dachxiu.chicagobooth.edu/download/KFQMLE.pdf))
   Analyzes Gaussian quasi-maximum likelihood estimation from noisy asynchronous
   observations via Kalman filtering and EM, proving consistency for integrated
   covariance under stochastic volatility despite misspecification of the
   constant-parameter working likelihood. Their fast factor specification treats
   observed noisy factors; exact state-space filtering with latent factors and
   historical data revisions requires Kalman smoothing passes over the
   observation history.

6. **Cappé, O. (2011)**, *Online EM Algorithm for Hidden Markov Models*, Journal
   of Computational and Graphical Statistics 20(3), 728–749.
   ([arXiv](https://arxiv.org/abs/0908.2359))
   Demonstrates that recursive online EM for latent-state models requires
   propagating auxiliary filtering-smoothing sufficient statistics forward in
   time; storing only filtered state means is insufficient to reproduce batch EM
   updates.

7. **Bibinger, M., N. Hautsch, P. Malec, and M. Reiß (2014, 2019)**, *Estimating
   the quadratic covariation matrix from noisy observations: Local method of
   moments and efficiency*, Annals of Statistics 42(4), 1312–1346
   ([arXiv:1303.6146](https://arxiv.org/abs/1303.6146)); and *Estimating the
   Spot Covariation of Asset Prices — Statistical Theory and Empirical Evidence*,
   Journal of Business & Economic Statistics 37(3), 419–435
   ([arXiv:1707.02419](https://arxiv.org/abs/1707.02419)).
   Projects asynchronous increments onto local orthogonal sine functions within
   blocks, applies noise bias correction per frequency, and combines frequencies
   with Fisher-information weights to achieve the semiparametric efficiency
   bound. Full multivariate efficiency weights couple all pairs through dense
   $`N^2 \times N^2`$ matrices, and the raw estimator is not constrained to be
   positive semidefinite.

8. **Li, D., O. B. Linton, and H. Zhang (2024)**, *Estimating Factor-Based
   Spot Volatility Matrices with Noisy and Asynchronous High-Frequency Data*,
   arXiv:2403.06246.
   ([arXiv](https://arxiv.org/abs/2403.06246))
   Combines local pre-averaging with principal-component factor extraction and
   POET residual thresholding for high-dimensional spot covariance matrices
   under assumptions of dense sampling across all assets and pervasive factors.

9. **Akahori, J., R. Kambara, N.-L. Liu, M. E. Mancino, T. Mariotti, and
   Y. Yasuda (2025)**, *Symmetric Positive Semi-Definite Fourier Estimator of
   Spot Covariance Matrix with High Frequency Data*, Risks 13(10), 197.
   ([DOI](https://doi.org/10.3390/risks13100197),
   [Preprint](https://www.preprints.org/manuscript/202505.0726))
   Constructs a positive-semidefinite spot covariance estimator directly from
   asynchronous returns using a positive-definite Fejér/Fourier frequency
   kernel, proving consistency in the noiseless Itô setting and evaluating
   noisy regimes via simulation.

10. **Zhao, D., and J. Chen (2026)**, *Factor Model Estimation for
    High-Dimensional Time Series with Heteroskedastic Noise*, arXiv:2610.00766.
    ([arXiv](https://arxiv.org/abs/2610.00766))
    Applies HeteroPCA to contemporaneous and lagged sample covariances on
    discrete panels under serially correlated factors (excluding white-noise
    factor increments). By contrast, `fit_factor` operates directly on
    continuous-time pairwise covariance rates over arbitrary sparse overlap
    graphs with explicit row-ball positive-definiteness constraints and
    structural graph diagnostics.

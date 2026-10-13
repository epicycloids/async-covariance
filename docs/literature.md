# Estimators and statistical targets

The implementation estimates a Brownian covariance **rate** from observations
with their actual timestamps. It does not identify an instantaneous covariance
matrix without a local-constant assumption. The raw pair estimates and the
positive-definite factor fit are separate outputs.

## Method comparison

| Method | Useful regime | Online work | Main restriction |
| --- | --- | --- | --- |
| HY rate | Asynchronous returns with little point noise; long-horizon rate estimation | Add or remove products of overlapping intervals | Raw products can have increasing variance as noisy observations become more frequent |
| Overlap slope | Constant-rate model, particularly imbalanced interval lengths | The same overlap index, with different product weights | Efficiency depends on the weights; pair-product dependence still matters |
| Pre-averaging / realised kernels | Noisy, frequently observed point prices | A recent observation buffer and lag products | Bandwidth, boundary terms, synchronisation, and the noise model are part of the estimator |
| Spectral local method of moments | Dense noisy observations; integrated or smoothed spot covariance | Block-wise sine coefficients and noise corrections | Efficient multivariate weights are dense; pilot estimates and smoothing couple blocks |
| PSD Fourier spot estimator | Asynchronous observations with a sufficiently small mesh | Add or remove fixed-frequency return coefficients | Frequencies, localization, horizon boundaries, and causal use require separate choices |
| Gaussian QML | Noisy point prices; likelihood-based reference calculations | Small pairwise Kalman filters, or a larger joint filter | Parameter updates and old corrections require replay or an explicit approximation |
| Local factor PCA | Large, sufficiently observed universes with locally stable factor structure | Block-wise denoising and subspace updates | Sampling density and smoothing assumptions are stronger than arbitrary sparse interval data |

The package's HY and overlap estimators are finite-sample estimating equations.
They should be compared on the same simulated paths. A kernel or QML method is
a separate method, not an alternative weighting silently applied to these
equations.

## The overlap equations

For asset $a$, let $r_e$ be its observed return over $I_e=[s_e,u_e)$,
$\ell_e=u_e-s_e$, and $o_{ef}=|I_e\cap I_f|$. Under constant drift $\mu$,
constant covariance rate $\Sigma$, and independent sampling times,

$$
\mathbb E[(r_e-\mu_a\ell_e)(r_f-\mu_b\ell_f)-\Gamma_{ef}]
=o_{ef}\Sigma_{ab}.
$$

Here $\Gamma$ is the covariance of the observation errors in the two returns.
The point errors are centered and independent of the latent process. For
independent point errors, the return error is the difference of its endpoint
errors. A shared endpoint contributes its variance with the product of the two
endpoint signs. In particular, a two-endpoint return has noise variance $2\tau_a^2$
and two successive returns have noise covariance $-\tau_a^2$ when point noise
variance is constant. Across assets, $\Gamma=0$ only under independent noise.

The two rate estimators use different coefficients:

$$
\widehat\Sigma^{\mathrm{HY}}_{ab}
=\frac{\sum w_{ef}(\widetilde r_e\widetilde r_f-\Gamma_{ef})}
       {\sum w_{ef}o_{ef}},\qquad
\widehat\Sigma^{\mathrm{slope}}_{ab}
=\frac{\sum w_{ef}h_{ef}o_{ef}(\widetilde r_e\widetilde r_f-\Gamma_{ef})}
       {\sum w_{ef}h_{ef}o_{ef}^2}.
$$

The sums include pairs with positive overlap. The package uses
$h_{ef}=1/(\ell_e\ell_f)$ and weights determined by timestamps. With exogenous
sampling and supplied true drift and noise variance, these ratios are unbiased
conditional on their observed timestamps. For noiseless, nonoverlapping
increment families, this choice of $h$ minimizes the variance within this class
of unbiased linear product estimators when $\Sigma_{ab}=0$. It is not a general
inverse-variance formula; estimating drift or choosing weights from the same
returns changes the finite-sample argument.

For centered Gaussian increment vectors $x,y$, a product statistic $S=x^TWy$
has variance

$$
\operatorname{Var}(S)
=\operatorname{tr}(W^TAWB)+\operatorname{tr}((W^TC)^2),
\quad A=\operatorname{Cov}(x),\ B=\operatorname{Cov}(y),\ C=\operatorname{Cov}(x,y).
$$

This identity follows from Gaussian fourth moments. Divide by the squared
estimator denominator for its model-based variance. A denominator or overlap
count alone is not an effective sample size. Overlapping intervals, reused
endpoints, and correlated pair products affect uncertainty even when the moment
equation is unbiased.

## Time weights, noise, and missing support

An exponential half-life $H$ means decay rate $\log(2)/H$. A fixed anchor for each
pair permits exact addition, removal, and decay of that pair's contribution.
With time-varying covariance, the expected product is
$\int_{I_e\cap I_f}\Sigma_{ab}(t)\,dt$; the two estimators consequently average
different time kernels, usually different for each asset pair. A factor fit does
not make those targets identical. Never shorten an overlap at a window boundary
while retaining the full observed return product.

Independent point noise need not bias a cross-asset product, but it can destroy
precision. For synchronous observations at spacing $T/n$, with independent point
noise variances $\tau_a^2,\tau_b^2$, the raw cross-product rate estimator has

$$
\operatorname{Var}\left(T^{-1}\sum_i r_{a,i}r_{b,i}\right)
\sim 6n\tau_a^2\tau_b^2/T^2.
$$

Correcting diagonal bias does not fix this divergence. Deterministic time-bin
coarsening limits points per bucket and preserves actual selected timestamps.
It does not impose a minimum elapsed return length: selected points can be close
across a bucket boundary. It is a sampling tradeoff, not pre-averaging or a
high-frequency consistency result. Proper pre-averaging also changes the
covariance exposure: for $R_e=\sum_j a_jr_j$, use $g_e(t)=\sum_j a_j1_{I_j}(t)$ and
$\int g_e(t)g_f(t)\,dt$, plus the corresponding endpoint-noise coefficients.

Pairs without overlap have no direct covariance estimate. They remain missing,
with zero fitting weight. A low-rank model can predict them, but graph
connectivity alone does not identify those predictions: each row needs enough
informative directions, and disconnected components have undetermined relative
rotations. The factor output is a regularized model fitted to observed moments.
It is not a global-optimality or identification certificate.

## Primary references and implementation consequences

1. [Hayashi and Yoshida (2005), *On covariance estimation of non-synchronously
   observed diffusion processes*](https://www.ms.u-tokyo.ac.jp/~nakahiro/mypapers_for_personal_use/hayyos03.pdf).
   The overlap-product estimator targets integrated covariance. Its sampling
   assumptions and target should be distinguished from a constant-rate slope.
2. [Christensen, Podolskij, and Vetter (2013), *On covariation estimation for
   multivariate continuous Itô semimartingales with noise in non-synchronous
   observation schemes*](https://doi.org/10.1016/j.jmva.2013.05.002),
   [institutional preprint](https://eldorado.tu-dortmund.de/bitstreams/ad4155d5-17c4-4666-a8f6-e2f4edc3746b/download).
   Pre-averaged HY works directly on asynchronous noisy points. Its averaging
   weights, growing bandwidth, normalization, and overlap rule define the method.
3. [Barndorff-Nielsen, Hansen, Lunde, and Shephard (2011), *Multivariate realised
   kernels*](https://doi.org/10.1016/j.jeconom.2010.07.009),
   [institutional preprint](https://pure.au.dk/ws/files/14416443/rp08_63.pdf).
   Refresh sampling and a suitable common lag kernel provide a PSD integrated
   covariance estimator. Pair-specific refresh schedules or normalization lose
   that common-matrix argument.
4. [Holý and Tomanová, *Streaming Approach to Quadratic Covariation Estimation
   Using Financial Ultra-High-Frequency Data*, v3 (2021)](https://arxiv.org/abs/2003.13062v3).
   Section 3.2 gives fixed-bandwidth quadratic-form recurrences with
   $O(N^2+NH)$ storage. The construction neglects boundary effects and trades
   statistical bandwidth adaptation for bounded memory.
5. [Shephard and Xiu (2017), *Econometric analysis of multivariate realised
   QML*](https://doi.org/10.1016/j.jeconom.2017.04.003),
   [author PDF](https://dachxiu.chicagobooth.edu/download/KFQMLE.pdf).
   Provides noisy asynchronous likelihood and EM/QML comparisons. Its fast factor
   treatment focuses on observed noisy factors; this is not a complexity result
   for an exact large latent-factor filter.
6. [Cappé (2011), *Online EM Algorithm for Hidden Markov Models*](https://arxiv.org/abs/0908.2359).
   Recursive EM needs smoothing sufficient statistics. A current latent mean or
   a frozen factor archive does not reproduce batch EM.
7. [Li, Linton, and Zhang (2024), *Estimating Factor-Based Spot Volatility
   Matrices with Noisy and Asynchronous High-Frequency Data*](https://arxiv.org/abs/2403.06246).
   Local pre-averaging, factor PCA, and POET offer a high-dimensional alternative
   under explicit local-stationarity, density, noise, and sparsity assumptions.
8. [Bibinger, Hautsch, Malec, and Reiß (2014), *Estimating the quadratic
   covariation matrix from noisy observations: Local method of moments and
   efficiency*](https://arxiv.org/abs/1303.6146),
   [spot extension, published 2019](https://arxiv.org/abs/1707.02419).
   Noise-corrected spectral products with Fisher-information weights provide an
   efficiency reference. The common block size depends on the least frequently
   observed process. Full multivariate weights act on vectorized matrices and
   are dense; independent scalar pair weights do not attain the same efficiency
   claim. Raw output is not guaranteed PSD. The spot extension estimates noise
   autocorrelation, with additional assumptions on its interaction with sampling.
9. [Akahori, Kambara, Liu, Mancino, Mariotti, and Yasuda (2025), *Symmetric
   Positive Semi-Definite Fourier Estimator of Spot Covariance Matrix with High
   Frequency Data*](https://doi.org/10.3390/risks13100197),
   [author preprint](https://www.preprints.org/manuscript/202505.0726).
   A common positive-definite frequency kernel yields a PSD spot estimate on
   asynchronous grids. The consistency theorem analyzes the latent Itô process;
   the noisy-model comparisons are simulation results with auxiliary parameter
   optimization. They do not establish arbitrary-noise consistency or a
   real-time boundary contract.

## Consequences of the source check through October 2026

For dense noisy prices, pre-averaged HY and spectral LMM are useful next
benchmarks. They address the variance problem that diagonal correction leaves
unsolved. The PSD Fourier construction is also worth comparing, especially
when its fixed-frequency coefficients permit local return corrections. Its
coefficients require storage proportional to assets times frequencies, in
addition to retained observations; materializing the covariance and choosing
causal boundary weights remain separate work. Changing a frequency basis or
horizon requires recomputation. These are alternative statistical contracts,
not guarantees inherited by the current overlap engine.

Recent panel-factor results do not by themselves identify a factor completion
from an arbitrary sparse overlap graph. For example,
[Zhao and Chen (September 2026)](https://arxiv.org/abs/2610.00766) combine
contemporaneous and lagged covariances with HeteroPCA on a panel; their model
excludes white-noise factors, unlike constant-rate Brownian return factors.
Density, factor strength, residual sparsity, and noise assumptions must be
checked before transferring such results. A newer upload date is also not a new method:
[arXiv:2602.19645](https://arxiv.org/abs/2602.19645) explicitly links to the
Christensen–Kinnebrock–Podolskij paper published in 2010.

`benchmarks/quality.py` measures stationary rate error, sampling imbalance, and
point-noise infill with and without time-bin coarsening. Its Monte Carlo standard
errors describe uncertainty from finite replication; they are not confidence
intervals returned by the estimator.

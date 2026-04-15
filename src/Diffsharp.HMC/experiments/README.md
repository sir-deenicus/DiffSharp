# Experiments

These scripts are sketches, not productized APIs.

`HmcKernel.fsx` keeps the core sampler deliberately small:

- HMC only needs a differentiable log-density over a vector state.
- DiffSharp already gives us `dsharp.grad`, tensor arithmetic, and random draws.
- That means a basic HMC kernel is mostly leapfrog integration plus a Metropolis accept/reject step.
- The current sketch now includes warmup adaptation for step size and a diagonal mass matrix, so examples can start from rough initial settings and settle into a better Euclidean geometry before sampling.
- There is also now a separate dynamic-HMC path with no-U-turn termination, multinomial proposal selection over the built trajectory, and diagnostics for divergences and tree depth.

`hmc_linear_regression.fsx` shows the direct path: write a log posterior over a parameter vector and sample it.

`hmc_high_dim_linear_regression.fsx` is the same direct style scaled up to an intercept plus a 12-dimensional coefficient vector, so the state is meaningfully higher-dimensional without introducing extra DSL machinery.

`hmc_high_dim_linear_regression_ce.fsx` expresses that same higher-dimensional regression through the computation-expression front-end, using a scalar latent for the intercept, a vector latent for the weights, and a vectorized observation, and it now drives the dynamic-HMC path.

`stan_like_ce.fsx` now sketches a lighter probabilistic-programming front-end rather than a Stan clone:

- `let!` introduces latent continuous parameters.
- `do!` adds factors or observations.
- `and!` can be used to declare independent latents without extra boilerplate.
- Compilation still lowers the model to an unconstrained parameter vector plus transforms and log-Jacobian corrections.
- The compiled model exposes a plain `Tensor -> Tensor` log-density, which is exactly the interface HMC wants.

That is the main design point if we want a real spec language here: keep the CE as a front-end that feels like a small probabilistic program, then compile it to `logDensity`, parameter pack/unpack, transforms, and diagnostics.

Typical next extensions would be:

- richer constraints such as simplex, lower and upper bounds, and covariance factors
- transformed parameters and generated quantities blocks
- denser warmup machinery such as windowed adaptation and dense mass matrices
- more sampler diagnostics and validation against reference implementations

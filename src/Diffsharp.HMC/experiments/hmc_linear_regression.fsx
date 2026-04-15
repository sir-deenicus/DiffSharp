#load "HmcKernel.fsx"

open DiffSharp
open HmcKernel

dsharp.config(backend=Backend.Reference, dtype=Dtype.Float64, device=Device.CPU)
dsharp.seed(7)

let scalar value = dsharp.tensor(value)
let logSqrtTwoPi = scalar (0.5 * log (2.0 * System.Math.PI))

let normalLogProb (mean: Tensor) (stddev: Tensor) (value: Tensor) =
    let z = (value - mean) / stddev
    (-0.5 * (z.pow(2.0))) - dsharp.log(stddev) - logSqrtTwoPi

let x = dsharp.tensor([ for i in -10 .. 10 -> float i / 5.0 ])
let trueAlpha = scalar 1.25
let trueBeta = scalar -0.8
let observationNoise = scalar 0.35
let y = trueAlpha + trueBeta * x + observationNoise * dsharp.randnLike(x)

let logPosterior (theta: Tensor) =
    let alpha = theta[0]
    let beta = theta[1]
    let mean = alpha + beta * x

    let alphaPrior = normalLogProb (scalar 0.0) (scalar 2.0) alpha
    let betaPrior = normalLogProb (scalar 0.0) (scalar 2.0) beta
    let likelihood = normalLogProb mean observationNoise y |> dsharp.sum

    alphaPrior + betaPrior + likelihood

let config: HmcKernel.Config =
    {
        StepSize = 0.04
        LeapfrogSteps = 20
        BurnIn = 500
        SampleCount = 1000
        Mass = None
        Adaptation = Some HmcKernel.defaultAdaptation
    }

let initialPosition = dsharp.tensor([ 0.0; 0.0 ])
let chain = HmcKernel.sample logPosterior initialPosition config
let posteriorMean = HmcKernel.sampleMean chain
let posteriorSamples = HmcKernel.stackSamples chain
let centeredSamples = posteriorSamples - posteriorMean
let posteriorStd = dsharp.mean(centeredSamples * centeredSamples, 0).sqrt()

printfn "Acceptance rate: %.3f" chain.Diagnostics.AcceptanceRate
printfn "Posterior mean [alpha; beta]: %A" posteriorMean
printfn "Posterior std  [alpha; beta]: %A" posteriorStd
printfn "Ground truth   [alpha; beta]: %A" (dsharp.tensor([ float trueAlpha; float trueBeta ]))

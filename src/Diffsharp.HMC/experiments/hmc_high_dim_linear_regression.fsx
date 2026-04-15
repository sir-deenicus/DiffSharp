#load "HmcKernel.fsx"

open DiffSharp
open HmcKernel

dsharp.config(backend=Backend.Reference, dtype=Dtype.Float64, device=Device.CPU)
dsharp.seed(19)

let scalar value = dsharp.tensor(value)
let logSqrtTwoPi = scalar (0.5 * log (2.0 * System.Math.PI))

let normalLogProb (mean: Tensor) (stddev: Tensor) (value: Tensor) =
    let z = (value - mean) / stddev
    (-0.5 * (z.pow(2.0))) - dsharp.log(stddev) - logSqrtTwoPi

let sampleCount = 120
let featureCount = 12
let parameterCount = featureCount + 1

let x = dsharp.randn([sampleCount; featureCount])
let trueIntercept = scalar -0.35
let trueWeights =
    dsharp.tensor(
        [
            -1.10
            -0.80
            -0.45
            -0.10
            0.15
            0.35
            0.60
            0.85
            1.05
            0.70
            -0.55
            0.25
        ]
    )

let observationNoise = scalar 0.40
let y = trueIntercept + x.matmul(trueWeights) + observationNoise * dsharp.randn([sampleCount])

let logPosterior (theta: Tensor) =
    let intercept = theta[0]
    let weights = theta[1..featureCount]
    let mean = intercept + x.matmul(weights)

    let interceptPrior = normalLogProb (scalar 0.0) (scalar 1.5) intercept
    let weightsPrior = normalLogProb (scalar 0.0) (scalar 1.0) weights |> dsharp.sum
    let likelihood = normalLogProb mean observationNoise y |> dsharp.sum

    interceptPrior + weightsPrior + likelihood

let config: HmcKernel.Config =
    {
        StepSize = 0.015
        LeapfrogSteps = 28
        BurnIn = 700
        SampleCount = 1400
        Mass = None
        Adaptation = Some HmcKernel.defaultAdaptation
    }

let initialPosition = dsharp.tensor(Array.zeroCreate<float> parameterCount)
let chain = HmcKernel.sample logPosterior initialPosition config
let posteriorMean = HmcKernel.sampleMean chain
let posteriorSamples = HmcKernel.stackSamples chain
let centeredSamples = posteriorSamples - posteriorMean
let posteriorStd = dsharp.mean(centeredSamples * centeredSamples, 0).sqrt()
let groundTruth = dsharp.cat([ trueIntercept.view(-1); trueWeights ])

printfn "State dimension: %d" parameterCount
printfn "Acceptance rate: %.3f" chain.Diagnostics.AcceptanceRate
printfn "Posterior mean [intercept; weights...]: %A" posteriorMean
printfn "Posterior std  [intercept; weights...]: %A" posteriorStd
printfn "Ground truth   [intercept; weights...]: %A" groundTruth

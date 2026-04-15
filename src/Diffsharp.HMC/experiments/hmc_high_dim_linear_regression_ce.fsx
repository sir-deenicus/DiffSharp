#load "HmcKernel.fsx"
#load "hansei_like_ce.fsx"

open DiffSharp
open HmcKernel 
open Hansei_like_ce

dsharp.config(backend=Backend.Reference, dtype=Dtype.Float64, device=Device.CPU)
dsharp.seed(19)

let sampleCount = 120
let featureCount = 12

let xData = dsharp.randn([sampleCount; featureCount])
let trueIntercept = -0.35
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

let observationNoise = 0.40
let yData =
    scalar trueIntercept + xData.matmul(trueWeights) + scalar observationNoise * dsharp.randn([sampleCount])

let x = Term.data xData
let y = Term.data yData

let model =
    prob {
        let! intercept = latent "intercept" (dsharp.tensor(0.0)) (normal (Term.scalar 0.0) (Term.scalar 1.5))
        let! weights = latent "weights" (dsharp.tensor(Array.zeroCreate<float> featureCount)) (normal (Term.scalar 0.0) (Term.scalar 1.0))

        let mean = intercept + Term.matmul x weights
        do! observe "y" (normal mean (Term.scalar observationNoise)) y

        return Term.cat [ intercept; weights ]
    }

let config: HmcKernel.Config =
    {
        StepSize = 0.015
        LeapfrogSteps = 28
        BurnIn = 700
        SampleCount = 1400
        Mass = None
        Adaptation = Some HmcKernel.defaultAdaptation
    }

let chain = HmcKernel.sampleDynamic model.LogDensity model.InitialPosition config HmcKernel.defaultDynamic
let unconstrainedMean = HmcKernel.sampleMean chain
let posteriorMean = model.Project unconstrainedMean
let decodedMean = model.Decode unconstrainedMean
let groundTruth = dsharp.cat([ dsharp.tensor([ trueIntercept ]); trueWeights ])

printfn "State dimension: %d" (featureCount + 1)
printfn "Move rate: %.3f" chain.Diagnostics.AcceptanceRate
printfn "Average acceptance prob: %.3f" chain.Diagnostics.AverageAcceptanceProbability
printfn "Divergences: %d" chain.Diagnostics.Divergences
printfn "Mean tree depth: %.2f" chain.Diagnostics.MeanTreeDepth
printfn "Max tree depth hits: %d" chain.Diagnostics.MaxTreeDepthHits
printfn "Posterior mean [intercept; weights...]: %A" posteriorMean
printfn "Decoded intercept: %A" decodedMean.["intercept"]
printfn "Decoded weights:   %A" decodedMean.["weights"]
printfn "Ground truth:      %A" groundTruth

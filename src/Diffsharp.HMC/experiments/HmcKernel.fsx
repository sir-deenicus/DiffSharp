#r "../../../bundles/DiffSharp-lite/bin/Debug/net8.0/DiffSharp.Core.dll"
#r "../../../bundles/DiffSharp-lite/bin/Debug/net8.0/DiffSharp.Backends.Reference.dll"
#r "../../../bundles/DiffSharp-lite/bin/Debug/net8.0/DiffSharp.Backends.Torch.dll"
#r "../../../bundles/DiffSharp-lite/bin/Debug/net8.0/DiffSharp.Data.dll"

open System
open DiffSharp

dsharp.config(backend=Backend.Reference, dtype=Dtype.Float64, device=Device.CPU)

type Config =
    {
        StepSize: float
        LeapfrogSteps: int
        BurnIn: int
        SampleCount: int
        Mass: Tensor option
        Adaptation: AdaptationConfig option
    }

and AdaptationConfig =
    {
        AdaptStepSize: bool
        AdaptMass: bool
        TargetAcceptanceRate: float
        StepSizeGamma: float
        StepSizeT0: float
        StepSizeKappa: float
        MassRegularization: float
    }

type DynamicConfig =
    {
        MaxTreeDepth: int
        MaxDeltaEnergy: float
    }

type Diagnostics =
    {
        Accepted: int
        Rejected: int
        AcceptanceRate: float
        AverageAcceptanceProbability: float
        Divergences: int
        MaxTreeDepthHits: int
        MeanTreeDepth: float
        MaxObservedEnergyError: float
        FinalStepSize: float
        FinalMass: Tensor
    }

type Chain =
    {
        Positions: Tensor[]
        LogDensities: Tensor[]
        Diagnostics: Diagnostics
    }

type private DualAveragingState =
    {
        Iteration: int
        Mu: float
        LogStepSize: float
        LogStepSizeAverage: float
        ErrorAverage: float
    }

type private RunningVariance =
    {
        Count: int
        Mean: Tensor
        M2: Tensor
    }

type private PhasePoint =
    {
        Position: Tensor
        Momentum: Tensor
        LogDensity: Tensor
        Hamiltonian: float
    }

type private DynamicTree =
    {
        Left: PhasePoint
        Right: PhasePoint
        Proposal: PhasePoint
        LogWeight: float
        SumAcceptanceProbability: float
        ProposalCount: int
        Divergent: bool
        Turning: bool
        MaxEnergyError: float
        LeafCount: int
    }

let defaultAdaptation =
    {
        AdaptStepSize = true
        AdaptMass = true
        TargetAcceptanceRate = 0.8
        StepSizeGamma = 0.05
        StepSizeT0 = 10.0
        StepSizeKappa = 0.75
        MassRegularization = 1e-3
    }

let defaultDynamic =
    {
        MaxTreeDepth = 10
        MaxDeltaEnergy = 1000.0
    }

let private validateInitialPosition (position: Tensor) =
    if position.dim <> 1 then
        failwithf "HMC expects a vector-valued state, got shape %A" position.shape
    position

let private validateAdaptation (adaptation: AdaptationConfig) =
    if adaptation.TargetAcceptanceRate <= 0.0 || adaptation.TargetAcceptanceRate >= 1.0 then
        failwithf "TargetAcceptanceRate must be in (0, 1), got %f" adaptation.TargetAcceptanceRate

    if adaptation.StepSizeGamma <= 0.0 then
        failwithf "StepSizeGamma must be positive, got %f" adaptation.StepSizeGamma

    if adaptation.StepSizeT0 < 0.0 then
        failwithf "StepSizeT0 must be non-negative, got %f" adaptation.StepSizeT0

    if adaptation.StepSizeKappa <= 0.5 || adaptation.StepSizeKappa > 1.0 then
        failwithf "StepSizeKappa must be in (0.5, 1], got %f" adaptation.StepSizeKappa

    if adaptation.MassRegularization <= 0.0 then
        failwithf "MassRegularization must be positive, got %f" adaptation.MassRegularization

    adaptation

let private validateDynamic (dynamicConfig: DynamicConfig) =
    if dynamicConfig.MaxTreeDepth < 1 then
        failwithf "MaxTreeDepth must be positive, got %d" dynamicConfig.MaxTreeDepth

    if dynamicConfig.MaxDeltaEnergy <= 0.0 then
        failwithf "MaxDeltaEnergy must be positive, got %f" dynamicConfig.MaxDeltaEnergy

    dynamicConfig

let private resolveMass (position: Tensor) (mass: Tensor option) =
    match mass with
    | Some massTensor when massTensor.shape = position.shape -> massTensor
    | Some massTensor -> failwithf "Mass must match the state shape, got %A and %A" massTensor.shape position.shape
    | None -> position.onesLike()

let private sanitizeStepSize fallback candidate =
    if Double.IsNaN(candidate) || Double.IsInfinity(candidate) || candidate <= 1e-12 then
        fallback
    else
        candidate

let private acceptanceProbability logAcceptanceRatio =
    if Double.IsNaN(logAcceptanceRatio) then
        0.0
    elif logAcceptanceRatio >= 0.0 then
        1.0
    else
        exp logAcceptanceRatio

let private logSumExp left right =
    if Double.IsNegativeInfinity(left) then
        right
    elif Double.IsNegativeInfinity(right) then
        left
    else
        let m = max left right
        m + log (exp (left - m) + exp (right - m))

let private chooseByLogWeight leftWeight rightWeight =
    if Double.IsNegativeInfinity(rightWeight) then
        false
    elif Double.IsNegativeInfinity(leftWeight) then
        true
    else
        let total = logSumExp leftWeight rightWeight
        let probabilityRight = exp (rightWeight - total)
        float (dsharp.rand(1)[0]) < probabilityRight

let private initDualAveraging stepSize =
    let initial = log stepSize

    {
        Iteration = 0
        Mu = log (10.0 * stepSize)
        LogStepSize = initial
        LogStepSizeAverage = initial
        ErrorAverage = 0.0
    }

let private updateDualAveraging (adaptation: AdaptationConfig) acceptProb state =
    let iteration = state.Iteration + 1
    let eta = 1.0 / (float iteration + adaptation.StepSizeT0)
    let errorAverage =
        ((1.0 - eta) * state.ErrorAverage) + (eta * (adaptation.TargetAcceptanceRate - acceptProb))
    let logStepSize =
        state.Mu - ((sqrt (float iteration)) / adaptation.StepSizeGamma) * errorAverage
    let shrinkage = (float iteration) ** (-adaptation.StepSizeKappa)
    let logStepSizeAverage =
        (shrinkage * logStepSize) + ((1.0 - shrinkage) * state.LogStepSizeAverage)

    {
        Iteration = iteration
        Mu = state.Mu
        LogStepSize = logStepSize
        LogStepSizeAverage = logStepSizeAverage
        ErrorAverage = errorAverage
    }

let private currentDualStepSize fallback state =
    exp state.LogStepSize |> sanitizeStepSize fallback

let private finalDualStepSize fallback state =
    exp state.LogStepSizeAverage |> sanitizeStepSize fallback

let private initRunningVariance (position: Tensor) =
    {
        Count = 0
        Mean = position * 0.0
        M2 = position * 0.0
    }

let private updateRunningVariance (sample: Tensor) (stats: RunningVariance) =
    let count = stats.Count + 1
    let delta = sample - stats.Mean
    let mean = stats.Mean + delta / float count
    let delta2 = sample - mean
    let m2 = stats.M2 + (delta * delta2)

    {
        Count = count
        Mean = mean
        M2 = m2
    }

let private diagonalMassFromVariance (regularization: float) (fallback: Tensor) (stats: RunningVariance) =
    if stats.Count < 2 then
        fallback
    else
        (stats.M2 / float (stats.Count - 1)) + (fallback.onesLike() * regularization)

let private potentialEnergy logDensity position = -logDensity position

let private kineticEnergy mass momentum =
    0.5 * dsharp.sum((momentum * momentum) / mass)

let private hamiltonian logDensity mass position momentum =
    potentialEnergy logDensity position + kineticEnergy mass momentum

let private potentialGradient logDensity position =
    dsharp.grad (potentialEnergy logDensity) position

let private makePhasePoint logDensity mass position momentum =
    let logDensityValue = logDensity position

    {
        Position = position
        Momentum = momentum
        LogDensity = logDensityValue
        Hamiltonian = float (hamiltonian logDensity mass position momentum)
    }

let private leapfrogStep logDensity (stepSize: float) (mass: Tensor) direction (position: Tensor) (momentum: Tensor) =
    let signedStepSize = (float direction) * stepSize
    let pHalf = momentum - ((potentialGradient logDensity position) * (0.5 * signedStepSize))
    let qNext = position + ((pHalf / mass) * signedStepSize)
    let pNext = pHalf - ((potentialGradient logDensity qNext) * (0.5 * signedStepSize))
    qNext, pNext

let private isUTurn (left: PhasePoint) (right: PhasePoint) =
    let delta = right.Position - left.Position
    let leftDot = float (dsharp.sum(delta * left.Momentum))
    let rightDot = float (dsharp.sum(delta * right.Momentum))
    leftDot < 0.0 || rightDot < 0.0

let private moved (position: Tensor) (candidate: Tensor) =
    float (dsharp.sum((candidate - position).pow(2.0))) > 0.0

let rec private buildDynamicTree
    logDensity
    (stepSize: float)
    (mass: Tensor)
    (dynamicConfig: DynamicConfig)
    (startHamiltonian: float)
    direction
    depth
    (point: PhasePoint)
    =
    if depth = 0 then
        let nextPosition, nextMomentum =
            leapfrogStep logDensity stepSize mass direction point.Position point.Momentum

        let nextPoint = makePhasePoint logDensity mass nextPosition nextMomentum
        let energyError = nextPoint.Hamiltonian - startHamiltonian
        let divergent =
            Double.IsNaN(nextPoint.Hamiltonian)
            || Double.IsInfinity(nextPoint.Hamiltonian)
            || energyError > dynamicConfig.MaxDeltaEnergy

        let logWeight =
            if divergent then Double.NegativeInfinity
            else -nextPoint.Hamiltonian

        {
            Left = nextPoint
            Right = nextPoint
            Proposal = nextPoint
            LogWeight = logWeight
            SumAcceptanceProbability =
                if divergent then 0.0
                else acceptanceProbability (startHamiltonian - nextPoint.Hamiltonian)
            ProposalCount = 1
            Divergent = divergent
            Turning = false
            MaxEnergyError = energyError
            LeafCount = 1
        }
    else
        let firstHalf =
            buildDynamicTree logDensity stepSize mass dynamicConfig startHamiltonian direction (depth - 1) point

        if firstHalf.Divergent || firstHalf.Turning then
            firstHalf
        else
            let secondStart =
                if direction < 0 then firstHalf.Left else firstHalf.Right

            let secondHalf =
                buildDynamicTree logDensity stepSize mass dynamicConfig startHamiltonian direction (depth - 1) secondStart

            let leftPoint =
                if direction < 0 then secondHalf.Left else firstHalf.Left

            let rightPoint =
                if direction < 0 then firstHalf.Right else secondHalf.Right

            let turning =
                firstHalf.Turning
                || secondHalf.Turning
                || isUTurn leftPoint rightPoint

            let chooseSecond = chooseByLogWeight firstHalf.LogWeight secondHalf.LogWeight

            {
                Left = leftPoint
                Right = rightPoint
                Proposal = if chooseSecond then secondHalf.Proposal else firstHalf.Proposal
                LogWeight = logSumExp firstHalf.LogWeight secondHalf.LogWeight
                SumAcceptanceProbability = firstHalf.SumAcceptanceProbability + secondHalf.SumAcceptanceProbability
                ProposalCount = firstHalf.ProposalCount + secondHalf.ProposalCount
                Divergent = firstHalf.Divergent || secondHalf.Divergent
                Turning = turning
                MaxEnergyError = max firstHalf.MaxEnergyError secondHalf.MaxEnergyError
                LeafCount = firstHalf.LeafCount + secondHalf.LeafCount
            }

let private applyWarmupAdaptation
    (adaptation: AdaptationConfig option)
    burnIn
    step
    acceptProb
    currentPosition
    fallbackStepSize
    fallbackMass
    (stepSize: byref<float>)
    (mass: byref<Tensor>)
    (stepSizeState: byref<DualAveragingState option>)
    (massState: byref<RunningVariance option>)
    =
    if step < burnIn then
        match adaptation, stepSizeState with
        | Some settings, Some state ->
            let updated = updateDualAveraging settings acceptProb state
            stepSizeState <- Some updated

            stepSize <-
                if step = burnIn - 1 then finalDualStepSize fallbackStepSize updated
                else currentDualStepSize fallbackStepSize updated
        | _ -> ()

        match adaptation, massState with
        | Some settings, Some state ->
            let updated = updateRunningVariance currentPosition state
            massState <- Some updated

            if updated.Count >= 2 then
                mass <- diagonalMassFromVariance settings.MassRegularization fallbackMass updated
        | _ -> ()

let leapfrog logDensity (stepSize: float) leapfrogSteps (mass: Tensor) (position: Tensor) (momentum: Tensor) : Tensor * Tensor =
    if leapfrogSteps < 1 then
        failwith "LeapfrogSteps must be positive."

    let p0 = momentum - ((potentialGradient logDensity position) * (0.5 * stepSize))

    let rec integrate (step: int) (q: Tensor) (p: Tensor) : Tensor * Tensor =
        let qNext = q + ((p / mass) * stepSize)

        if step = leapfrogSteps - 1 then
            let pNext = p - ((potentialGradient logDensity qNext) * (0.5 * stepSize))
            qNext, -pNext
        else
            let pNext = p - ((potentialGradient logDensity qNext) * stepSize)
            integrate (step + 1) qNext pNext

    integrate 0 position p0

let sample logDensity (initialPosition: Tensor) (config: Config) =
    let initialPosition = validateInitialPosition initialPosition
    let adaptation =
        config.Adaptation
        |> Option.map validateAdaptation
        |> Option.filter (fun _ -> config.BurnIn > 0)

    let mutable mass = resolveMass initialPosition config.Mass
    let totalSteps = config.BurnIn + config.SampleCount

    let mutable accepted = 0
    let mutable rejected = 0
    let mutable acceptanceProbabilitySum = 0.0
    let mutable maxObservedEnergyError = 0.0
    let samples = ResizeArray<Tensor>()
    let logDensities = ResizeArray<Tensor>()

    let baseStepSize = config.StepSize
    let baseMass = mass
    let mutable stepSize = config.StepSize
    let mutable currentPosition = initialPosition
    let mutable currentLogDensity = logDensity currentPosition
    let mutable stepSizeAdaptationState =
        adaptation
        |> Option.bind (fun settings ->
            if settings.AdaptStepSize then Some(initDualAveraging stepSize) else None)
    let mutable massAdaptationState =
        adaptation
        |> Option.bind (fun settings ->
            if settings.AdaptMass then Some(initRunningVariance currentPosition) else None)

    for step = 0 to totalSteps - 1 do
        let currentMomentum = dsharp.randnLike(currentPosition) * mass.sqrt()
        let proposedPosition, proposedMomentum =
            leapfrog logDensity stepSize config.LeapfrogSteps mass currentPosition currentMomentum

        let proposedLogDensity = logDensity proposedPosition
        let currentHamiltonian = hamiltonian logDensity mass currentPosition currentMomentum
        let proposedHamiltonian = hamiltonian logDensity mass proposedPosition proposedMomentum
        let logAcceptanceRatio = float (currentHamiltonian - proposedHamiltonian)
        let acceptProb = acceptanceProbability logAcceptanceRatio
        let energyError = float proposedHamiltonian - float currentHamiltonian
        let uniformDraw = max 1e-12 (float (dsharp.rand(1)[0]))

        acceptanceProbabilitySum <- acceptanceProbabilitySum + acceptProb
        maxObservedEnergyError <- max maxObservedEnergyError energyError

        if Math.Log(uniformDraw) < logAcceptanceRatio then
            currentPosition <- proposedPosition
            currentLogDensity <- proposedLogDensity
            accepted <- accepted + 1
        else
            rejected <- rejected + 1

        applyWarmupAdaptation
            adaptation
            config.BurnIn
            step
            acceptProb
            currentPosition
            baseStepSize
            baseMass
            &stepSize
            &mass
            &stepSizeAdaptationState
            &massAdaptationState

        if step >= config.BurnIn then
            samples.Add(currentPosition)
            logDensities.Add(currentLogDensity)

    let totalTransitions = accepted + rejected
    let acceptanceRate =
        if totalTransitions = 0 then 0.0
        else float accepted / float totalTransitions
    let averageAcceptanceProbability =
        if totalTransitions = 0 then 0.0
        else acceptanceProbabilitySum / float totalTransitions

    {
        Positions = samples.ToArray()
        LogDensities = logDensities.ToArray()
        Diagnostics =
            {
                Accepted = accepted
                Rejected = rejected
                AcceptanceRate = acceptanceRate
                AverageAcceptanceProbability = averageAcceptanceProbability
                Divergences = 0
                MaxTreeDepthHits = 0
                MeanTreeDepth = 0.0
                MaxObservedEnergyError = maxObservedEnergyError
                FinalStepSize = stepSize
                FinalMass = mass
            }
    }

let sampleDynamic logDensity (initialPosition: Tensor) (config: Config) (dynamicConfig: DynamicConfig) =
    let initialPosition = validateInitialPosition initialPosition
    let adaptation =
        config.Adaptation
        |> Option.map validateAdaptation
        |> Option.filter (fun _ -> config.BurnIn > 0)

    let dynamicConfig = validateDynamic dynamicConfig
    let mutable mass = resolveMass initialPosition config.Mass
    let totalSteps = config.BurnIn + config.SampleCount

    let mutable accepted = 0
    let mutable rejected = 0
    let mutable divergenceCount = 0
    let mutable maxTreeDepthHits = 0
    let mutable totalTreeDepth = 0.0
    let mutable acceptanceProbabilitySum = 0.0
    let mutable maxObservedEnergyError = 0.0
    let samples = ResizeArray<Tensor>()
    let logDensities = ResizeArray<Tensor>()

    let baseStepSize = config.StepSize
    let baseMass = mass
    let mutable stepSize = config.StepSize
    let mutable currentPosition = initialPosition
    let mutable currentLogDensity = logDensity currentPosition
    let mutable stepSizeAdaptationState =
        adaptation
        |> Option.bind (fun settings ->
            if settings.AdaptStepSize then Some(initDualAveraging stepSize) else None)
    let mutable massAdaptationState =
        adaptation
        |> Option.bind (fun settings ->
            if settings.AdaptMass then Some(initRunningVariance currentPosition) else None)

    for step = 0 to totalSteps - 1 do
        let currentMomentum = dsharp.randnLike(currentPosition) * mass.sqrt()
        let startPoint = makePhasePoint logDensity mass currentPosition currentMomentum
        let startHamiltonian = startPoint.Hamiltonian

        let mutable tree =
            {
                Left = startPoint
                Right = startPoint
                Proposal = startPoint
                LogWeight = -startHamiltonian
                SumAcceptanceProbability = 0.0
                ProposalCount = 0
                Divergent = false
                Turning = false
                MaxEnergyError = 0.0
                LeafCount = 0
            }

        let mutable depth = 0

        while depth < dynamicConfig.MaxTreeDepth && not tree.Turning && not tree.Divergent do
            let direction =
                if float (dsharp.rand(1)[0]) < 0.5 then -1 else 1

            let subtreeStart =
                if direction < 0 then tree.Left else tree.Right

            let subtree =
                buildDynamicTree logDensity stepSize mass dynamicConfig startHamiltonian direction depth subtreeStart

            let leftPoint =
                if direction < 0 then subtree.Left else tree.Left

            let rightPoint =
                if direction < 0 then tree.Right else subtree.Right

            let turning = tree.Turning || subtree.Turning || isUTurn leftPoint rightPoint
            let chooseSubtree = chooseByLogWeight tree.LogWeight subtree.LogWeight

            tree <-
                {
                    Left = leftPoint
                    Right = rightPoint
                    Proposal = if chooseSubtree then subtree.Proposal else tree.Proposal
                    LogWeight = logSumExp tree.LogWeight subtree.LogWeight
                    SumAcceptanceProbability = tree.SumAcceptanceProbability + subtree.SumAcceptanceProbability
                    ProposalCount = tree.ProposalCount + subtree.ProposalCount
                    Divergent = tree.Divergent || subtree.Divergent
                    Turning = turning
                    MaxEnergyError = max tree.MaxEnergyError subtree.MaxEnergyError
                    LeafCount = tree.LeafCount + subtree.LeafCount
                }

            depth <- depth + 1

        let meanAcceptProb =
            if tree.ProposalCount = 0 then 1.0
            else tree.SumAcceptanceProbability / float tree.ProposalCount

        acceptanceProbabilitySum <- acceptanceProbabilitySum + meanAcceptProb
        totalTreeDepth <- totalTreeDepth + float depth
        maxObservedEnergyError <- max maxObservedEnergyError tree.MaxEnergyError

        if tree.Divergent then
            divergenceCount <- divergenceCount + 1

        if depth >= dynamicConfig.MaxTreeDepth then
            maxTreeDepthHits <- maxTreeDepthHits + 1

        if moved currentPosition tree.Proposal.Position then
            currentPosition <- tree.Proposal.Position
            currentLogDensity <- tree.Proposal.LogDensity
            accepted <- accepted + 1
        else
            rejected <- rejected + 1

        applyWarmupAdaptation
            adaptation
            config.BurnIn
            step
            meanAcceptProb
            currentPosition
            baseStepSize
            baseMass
            &stepSize
            &mass
            &stepSizeAdaptationState
            &massAdaptationState

        if step >= config.BurnIn then
            samples.Add(currentPosition)
            logDensities.Add(currentLogDensity)

    let totalTransitions = accepted + rejected
    let acceptanceRate =
        if totalTransitions = 0 then 0.0
        else float accepted / float totalTransitions
    let averageAcceptanceProbability =
        if totalTransitions = 0 then 0.0
        else acceptanceProbabilitySum / float totalTransitions
    let meanTreeDepth =
        if totalTransitions = 0 then 0.0
        else totalTreeDepth / float totalTransitions

    {
        Positions = samples.ToArray()
        LogDensities = logDensities.ToArray()
        Diagnostics =
            {
                Accepted = accepted
                Rejected = rejected
                AcceptanceRate = acceptanceRate
                AverageAcceptanceProbability = averageAcceptanceProbability
                Divergences = divergenceCount
                MaxTreeDepthHits = maxTreeDepthHits
                MeanTreeDepth = meanTreeDepth
                MaxObservedEnergyError = maxObservedEnergyError
                FinalStepSize = stepSize
                FinalMass = mass
            }
    }

let stackSamples (chain: Chain) = dsharp.stack(chain.Positions)

let sampleMean (chain: Chain) = stackSamples chain |> dsharp.mean

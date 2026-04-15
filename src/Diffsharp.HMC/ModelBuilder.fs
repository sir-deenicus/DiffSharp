namespace Diffsharp.HMC

open DiffSharp

module ModelBuilder =
    type Constraint =
        | Unconstrained
        | Positive

    type ParameterSpec =
        {
            Name: string
            Initial: Tensor
            Constraint: Constraint
        }

    type ModelEnv =
        {
            Parameters: Map<string, Tensor>
        }

    type ModelState =
        {
            Parameters: ParameterSpec list
            Factors: (string * (ModelEnv -> Tensor)) list
        }

    type CompiledModel =
        {
            InitialPosition: Tensor
            LogDensity: Tensor -> Tensor
            Project: Tensor -> Tensor
            Decode: Tensor -> Map<string, Tensor>
        }

    type Term =
        private
        | Term of (ModelEnv -> Tensor)

        member term.Eval(env: ModelEnv) =
            let (Term run) = term
            run env

        static member internal Lift(run: ModelEnv -> Tensor) =
            Term run

        static member Constant(value: Tensor) =
            Term(fun _ -> value)

        static member Parameter(name: string) =
            Term(fun env -> env.Parameters.[name])

        static member (+) (left: Term, right: Term) = Term.Lift(fun env -> left.Eval(env) + right.Eval(env))
        static member (+) (left: Term, right: float) = left + Term.Constant(dsharp.tensor(right))
        static member (+) (left: float, right: Term) = Term.Constant(dsharp.tensor(left)) + right
        static member (+) (left: Term, right: Tensor) = left + Term.Constant(right)
        static member (+) (left: Tensor, right: Term) = Term.Constant(left) + right

        static member (-) (left: Term, right: Term) = Term.Lift(fun env -> left.Eval(env) - right.Eval(env))
        static member (-) (left: Term, right: float) = left - Term.Constant(dsharp.tensor(right))
        static member (-) (left: float, right: Term) = Term.Constant(dsharp.tensor(left)) - right
        static member (-) (left: Term, right: Tensor) = left - Term.Constant(right)
        static member (-) (left: Tensor, right: Term) = Term.Constant(left) - right
        static member (~-) (term: Term) = Term.Lift(fun env -> -(term.Eval(env)))

        static member (*) (left: Term, right: Term) = Term.Lift(fun env -> left.Eval(env) * right.Eval(env))
        static member (*) (left: Term, right: float) = left * Term.Constant(dsharp.tensor(right))
        static member (*) (left: float, right: Term) = Term.Constant(dsharp.tensor(left)) * right
        static member (*) (left: Term, right: Tensor) = left * Term.Constant(right)
        static member (*) (left: Tensor, right: Term) = Term.Constant(left) * right

        static member (/) (left: Term, right: Term) = Term.Lift(fun env -> left.Eval(env) / right.Eval(env))
        static member (/) (left: Term, right: float) = left / Term.Constant(dsharp.tensor(right))
        static member (/) (left: float, right: Term) = Term.Constant(dsharp.tensor(left)) / right
        static member (/) (left: Term, right: Tensor) = left / Term.Constant(right)
        static member (/) (left: Tensor, right: Term) = Term.Constant(left) / right

    module Term =
        let scalar value = Term.Constant(dsharp.tensor(value))
        let data (value: Tensor) = Term.Constant(value)
        let eval env (term: Term) = term.Eval(env)
        let sumAll (term: Term) = Term.Lift(fun env -> dsharp.sum(term.Eval(env)))
        let log (term: Term) = Term.Lift(fun env -> dsharp.log(term.Eval(env)))
        let exp (term: Term) = Term.Lift(fun env -> dsharp.exp(term.Eval(env)))
        let sigmoid (term: Term) = Term.Lift(fun env -> dsharp.sigmoid(term.Eval(env)))
        let pow power (term: Term) = Term.Lift(fun env -> term.Eval(env).pow(power))
        let softplus term = log (1.0 + exp term)
        let stack (terms: Term list) = Term.Lift(fun env -> terms |> List.map (eval env) |> dsharp.stack)
        let matmul (left: Term) (right: Term) = Term.Lift(fun env -> (eval env left).matmul(eval env right))
        let cat (terms: Term list) = Term.Lift(fun env -> terms |> List.map (fun term -> (eval env term).flatten()) |> dsharp.cat)

    type Distribution = private Distribution of (Term -> Term)

    module Distribution =
        let logProb (Distribution f) value = f value

    type ModelExpr<'a> = ModelState -> 'a * ModelState

    let scalar value = dsharp.tensor(value)

    let normal (mean: Term) (stddev: Term) =
        Distribution(fun value ->
            Term.Lift(fun env ->
                let meanValue = Term.eval env mean
                let stddevValue = Term.eval env stddev
                let valueTensor = Term.eval env value
                let z = (valueTensor - meanValue) / stddevValue
                (-0.5 * z.pow(2.0)) - dsharp.log(stddevValue) - scalar (0.5 * log (2.0 * System.Math.PI))))

    let private constrainedValue constraintKind raw =
        match constraintKind with
        | Unconstrained -> raw, scalar 0.0
        | Positive ->
            let value = dsharp.log(1.0 + dsharp.exp(raw))
            let logJacobian = dsharp.log(dsharp.sigmoid(raw)) |> dsharp.sum
            value, logJacobian

    let private appendParameter spec state =
        if state.Parameters |> List.exists (fun existing -> existing.Name = spec.Name) then
            failwithf "Duplicate parameter name '%s'." spec.Name
        { state with Parameters = state.Parameters @ [ spec ] }

    let private appendFactor name factor state =
        { state with Factors = state.Factors @ [ name, factor ] }

    let private evalScalarTerm context env term =
        let value = Term.eval env term

        if value.nelement <> 1 then
            failwithf
                "%s expected a scalar log score, got shape %A. Reduce explicitly before calling factor."
                context
                value.shape

        value

    let factor name (logWeight: Term) : ModelExpr<unit> =
        fun state ->
            (), appendFactor name (fun env -> evalScalarTerm (sprintf "factor '%s'" name) env logWeight) state

    let factorMany name (logWeights: Term) : ModelExpr<unit> =
        factor name (Term.sumAll logWeights)

    let observe name distribution value =
        factorMany name (Distribution.logProb distribution value)

    let latentWith constraintKind name initial distribution : ModelExpr<Term> =
        fun state ->
            let spec =
                {
                    Name = name
                    Initial = initial
                    Constraint = constraintKind
                }

            let term = Term.Parameter(name)
            let state = appendParameter spec state
            let (), state = factorMany (name + " prior") (Distribution.logProb distribution term) state
            term, state

    let latent name initial distribution = latentWith Unconstrained name initial distribution
    let positiveLatent name initial distribution = latentWith Positive name initial distribution

    let normalLatent name initial mean stddev =
        latent name (scalar initial) (normal (Term.scalar mean) (Term.scalar stddev))

    let positiveNormalLatent name initial mean stddev =
        positiveLatent name (scalar initial) (normal (Term.scalar mean) (Term.scalar stddev))

    let compileModel (program: ModelExpr<Term>) =
        let returnedTerm, state = program { Parameters = []; Factors = [] }

        let initialPosition =
            state.Parameters
            |> List.map (fun spec -> spec.Initial.flatten())
            |> dsharp.cat

        let decodeWithJacobian (position: Tensor) =
            let mutable offset = 0
            let mutable totalLogJacobian = scalar 0.0

            let parameters =
                state.Parameters
                |> List.map (fun spec ->
                    let count = spec.Initial.nelement
                    let raw = position[offset .. offset + count - 1].viewAs(spec.Initial)
                    offset <- offset + count

                    let value, logJacobian = constrainedValue spec.Constraint raw
                    totalLogJacobian <- totalLogJacobian + logJacobian
                    spec.Name, value)
                |> Map.ofList

            parameters, totalLogJacobian

        let decode position = decodeWithJacobian position |> fst

        let logDensity position =
            let parameters, totalLogJacobian = decodeWithJacobian position
            let env = { Parameters = parameters }
            state.Factors
            |> List.fold (fun acc (_, logFactor) -> acc + logFactor env) totalLogJacobian

        let project position =
            let parameters = decode position
            let env = { Parameters = parameters }
            Term.eval env returnedTerm

        {
            InitialPosition = initialPosition
            LogDensity = logDensity
            Project = project
            Decode = decode
        }

    type ProbModelBuilder() =
        member _.Return(value) : ModelExpr<'a> =
            fun state -> value, state

        member _.ReturnFrom(expr: ModelExpr<'a>) = expr

        member _.Bind(expr: ModelExpr<'a>, binder: 'a -> ModelExpr<'b>) : ModelExpr<'b> =
            fun state ->
                let value, state = expr state
                binder value state

        member _.BindReturn(expr: ModelExpr<'a>, binder: 'a -> 'b) : ModelExpr<'b> =
            fun state ->
                let value, state = expr state
                binder value, state

        member _.Zero() : ModelExpr<unit> =
            fun state -> (), state

        member _.Delay(generator: unit -> ModelExpr<'a>) : ModelExpr<'a> =
            fun state -> generator() state

        member _.Combine(left: ModelExpr<unit>, right: ModelExpr<'a>) : ModelExpr<'a> =
            fun state ->
                let (), state = left state
                right state

        member _.For(sequence: seq<'a>, body: 'a -> ModelExpr<unit>) : ModelExpr<unit> =
            fun state ->
                use enumerator = sequence.GetEnumerator()
                let mutable currentState = state

                while enumerator.MoveNext() do
                    let (), nextState = body enumerator.Current currentState
                    currentState <- nextState

                (), currentState

        member _.MergeSources(left: ModelExpr<'a>, right: ModelExpr<'b>) : ModelExpr<'a * 'b> =
            fun state ->
                let leftValue, state = left state
                let rightValue, state = right state
                (leftValue, rightValue), state

        member _.Run(program: ModelExpr<Term>) =
            compileModel program

    let prob = ProbModelBuilder()

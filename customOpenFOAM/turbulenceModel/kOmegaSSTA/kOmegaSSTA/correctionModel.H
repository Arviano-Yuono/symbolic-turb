// ------------------------------------------------------------------------
// correctionModel.H
//
// Recursive-descent evaluator for SpaRTA-style anisotropy correction:
//
//   TensorExpr := sum of (ScalarExpr * Tj) terms, with + / -
//   ScalarExpr uses I1, I2, numbers, parentheses, + - * / ^
//
// Restrictions (by design):
//   - No tensor-tensor multiplication (only scalar scaling of tensors)
//   - A "bare" scalar in TensorExpr is an error (must multiply a tensor)
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// Utilities
// ------------------------------------------------------------------------
inline volScalarField makeUniformScalar
(
    const fvMesh& mesh,
    const scalar val,
    const word& name = "srConstS"
)
{
    return volScalarField
    (
        IOobject
        (
            name,
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false   // do not register in objectRegistry
        ),
        mesh,
        dimensionedScalar(name, dimless, val)
    );
}

inline volSymmTensorField makeZeroSymmTensor
(
    const fvMesh& mesh,
    const word& name = "srZeroT"
)
{
    return volSymmTensorField
    (
        IOobject
        (
            name,
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        mesh,
        dimensionedSymmTensor(name, dimless, symmTensor::zero)
    );
}

// ========================================================================
// Scalar parser (for coefficients f(I1,I2))
// ========================================================================
inline bool isTensorMarker(const wordList& tokens, label i)
{
    return (i + 1 < tokens.size()
        && (tokens[i] == "*" || tokens[i] == "/")
        && (tokens[i+1] == "T1" || tokens[i+1] == "T2" || tokens[i+1] == "T3"));
}

inline bool isStopToken(const wordList& tokens, label i)
{
    if (i >= tokens.size()) return true;
    return tokens[i] == ")" || tokens[i] == "+" || tokens[i] == "-"
        || isTensorMarker(tokens, i);
}

inline void expectToken
(
    const wordList& tokens,
    const label i,
    const word& expected,
    const char* context
)
{
    if (i >= tokens.size() || tokens[i] != expected)
    {
        FatalErrorInFunction
            << "Expected '" << expected << "' in " << context << ", got '"
            << (i < tokens.size() ? tokens[i] : "<eof>") << "'" << nl
            << abort(FatalError);
    }
}

inline void consumeToken
(
    const wordList& tokens,
    label& i,
    const word& expected,
    const char* context
)
{
    expectToken(tokens, i, expected, context);
    ++i;
}

volScalarField evalScalarAddSub
(
    const wordList& tokens,
    label& i,
    const volScalarField& I1,
    const volScalarField& I2,
    const fvMesh& mesh,
    const volScalarField* N1Ptr = nullptr,
    const volScalarField* N2Ptr = nullptr,
    const volScalarField* N3Ptr = nullptr
);

volScalarField evalScalarFactor
(
    const wordList& tokens,
    label& i,
    const volScalarField& I1,
    const volScalarField& I2,
    const fvMesh& mesh,
    const volScalarField* N1Ptr,
    const volScalarField* N2Ptr,
    const volScalarField* N3Ptr
)
{
    if (i >= tokens.size())
    {
        FatalErrorInFunction
            << "Unexpected end of tokens in evalScalarFactor" << nl
            << abort(FatalError);
    }

    if (tokens[i] == "(")
    {
        ++i;
        volScalarField result = evalScalarAddSub
        (
            tokens, i, I1, I2, mesh, N1Ptr, N2Ptr, N3Ptr
        );
        consumeToken(tokens, i, ")", "scalar expression");
        return result;
    }

    if (tokens[i] == "I1") { i++; return I1; }
    if (tokens[i] == "I2") { i++; return I2; }
    if (tokens[i] == "N1")
    {
        if (!N1Ptr)
        {
            FatalErrorInFunction
                << "Token N1 is not available in this expression context" << nl
                << abort(FatalError);
        }
        i++;
        return *N1Ptr;
    }
    if (tokens[i] == "N2")
    {
        if (!N2Ptr)
        {
            FatalErrorInFunction
                << "Token N2 is not available in this expression context" << nl
                << abort(FatalError);
        }
        i++;
        return *N2Ptr;
    }
    if (tokens[i] == "N3")
    {
        if (!N3Ptr)
        {
            FatalErrorInFunction
                << "Token N3 is not available in this expression context" << nl
                << abort(FatalError);
        }
        i++;
        return *N3Ptr;
    }

    // Number (supports "-0.1" as a single token)
    scalar val = readScalar(IStringStream(tokens[i])());
    i++;
    return makeUniformScalar(mesh, val);
}

volScalarField evalScalarPower
(
    const wordList& tokens,
    label& i,
    const volScalarField& I1,
    const volScalarField& I2,
    const fvMesh& mesh,
    const volScalarField* N1Ptr,
    const volScalarField* N2Ptr,
    const volScalarField* N3Ptr
)
{
    volScalarField lhs = evalScalarFactor
    (
        tokens, i, I1, I2, mesh, N1Ptr, N2Ptr, N3Ptr
    );

    while (i < tokens.size() && tokens[i] == "^")
    {
        i++; // eat '^'
        if (i >= tokens.size())
        {
            FatalErrorInFunction
                << "Missing exponent after '^'" << nl
                << abort(FatalError);
        }

        scalar exponent = readScalar(IStringStream(tokens[i])());
        i++;
        lhs = pow(lhs, exponent);
    }

    return lhs;
}

volScalarField evalScalarMulDiv
(
    const wordList& tokens,
    label& i,
    const volScalarField& I1,
    const volScalarField& I2,
    const fvMesh& mesh,
    const volScalarField* N1Ptr,
    const volScalarField* N2Ptr,
    const volScalarField* N3Ptr
)
{
    volScalarField lhs = evalScalarPower
    (
        tokens, i, I1, I2, mesh, N1Ptr, N2Ptr, N3Ptr
    );

    while (i < tokens.size())
    {
        if (isStopToken(tokens, i))
        {
            break;
        }

        const word op = tokens[i];
        if (op != "*" && op != "/")
        {
            break;
        }
        ++i;

        if (op == "*")
        {
            lhs *= evalScalarPower(tokens, i, I1, I2, mesh, N1Ptr, N2Ptr, N3Ptr);
        }
        else
        {
            lhs /= evalScalarPower(tokens, i, I1, I2, mesh, N1Ptr, N2Ptr, N3Ptr);
        }
    }
    return lhs;
}

volScalarField evalScalarAddSub
(
    const wordList& tokens,
    label& i,
    const volScalarField& I1,
    const volScalarField& I2,
    const fvMesh& mesh,
    const volScalarField* N1Ptr,
    const volScalarField* N2Ptr,
    const volScalarField* N3Ptr
)
{
    volScalarField lhs = evalScalarMulDiv
    (
        tokens, i, I1, I2, mesh, N1Ptr, N2Ptr, N3Ptr
    );

    while (i < tokens.size())
    {
        const word op = tokens[i];
        if (op != "+" && op != "-")
        {
            break;
        }
        ++i;

        if (op == "+")
        {
            lhs += evalScalarMulDiv(tokens, i, I1, I2, mesh, N1Ptr, N2Ptr, N3Ptr);
        }
        else
        {
            lhs -= evalScalarMulDiv(tokens, i, I1, I2, mesh, N1Ptr, N2Ptr, N3Ptr);
        }
    }

    return lhs;
}

// ========================================================================
// Tensor parser (for full correction expression)
// ========================================================================

volSymmTensorField evalTensorAddSub
(
    const wordList& tokens,
    label& i,
    const volScalarField& I1,
    const volScalarField& I2,
    const volSymmTensorField& T1,
    const volSymmTensorField& T2,
    const volSymmTensorField& T3,
    const fvMesh& mesh
);

volSymmTensorField evalTensorFactor
(
    const wordList& tokens,
    label& i,
    const volScalarField& I1,
    const volScalarField& I2,
    const volSymmTensorField& T1,
    const volSymmTensorField& T2,
    const volSymmTensorField& T3,
    const fvMesh& mesh
)
{
    if (i >= tokens.size())
    {
        FatalErrorInFunction
            << "Unexpected end of tokens in evalTensorFactor" << nl
            << abort(FatalError);
    }

    if (tokens[i] == "(")
    {
        ++i;
        volSymmTensorField result = evalTensorAddSub(tokens, i, I1, I2, T1, T2, T3, mesh);
        consumeToken(tokens, i, ")", "tensor expression");
        return result;
    }

    if (tokens[i] == "T1") { i++; return T1; }
    if (tokens[i] == "T2") { i++; return T2; }
    if (tokens[i] == "T3") { i++; return T3; }

    FatalErrorInFunction
        << "Expected tensor token (T1/T2/T3 or '('), got '" << tokens[i] << "'" << nl
        << abort(FatalError);

    return makeZeroSymmTensor(mesh);
}

volSymmTensorField evalTensorTerm
(
    const wordList& tokens,
    label& i,
    const volScalarField& I1,
    const volScalarField& I2,
    const volSymmTensorField& T1,
    const volSymmTensorField& T2,
    const volSymmTensorField& T3,
    const fvMesh& mesh
)
{
    if (i >= tokens.size())
    {
        FatalErrorInFunction
            << "Unexpected end of tokens in evalTensorTerm" << nl
            << abort(FatalError);
    }

    bool wrapped = false;
    if (tokens[i] == "(")
    {
        wrapped = true;
        ++i;
    }

    // Parse scalar part until we hit "* Tj" (or "/ Tj")
    volScalarField s = evalScalarAddSub(tokens, i, I1, I2, mesh);

    consumeToken(tokens, i, "*", "tensor term before tensor basis");

    volSymmTensorField t = evalTensorFactor(tokens, i, I1, I2, T1, T2, T3, mesh);

    volSymmTensorField out = t * s;

    if (wrapped)
    {
        consumeToken(tokens, i, ")", "tensor term");
    }

    return out;
}


volSymmTensorField evalTensorMulDiv
(
    const wordList& tokens,
    label& i,
    const volScalarField& I1,
    const volScalarField& I2,
    const volSymmTensorField& T1,
    const volSymmTensorField& T2,
    const volSymmTensorField& T3,
    const fvMesh& mesh
)
{
    volSymmTensorField lhs = evalTensorTerm(tokens, i, I1, I2, T1, T2, T3, mesh);

    while (i < tokens.size())
    {
        // Stop at add/sub or closing paren, handled by higher level
        if (tokens[i] == "+" || tokens[i] == "-" || tokens[i] == ")") break;

        // Allow only scaling by a scalar expr: (lhs * scalar) or (lhs / scalar)
        const word op = tokens[i];
        if (op == "*" || op == "/")
        {
            ++i;

            volScalarField s = evalScalarAddSub(tokens, i, I1, I2, mesh);

            if (op == "*") lhs = lhs * s;
            else           lhs = lhs / s;
        }
        else
        {
            break;
        }
    }

    return lhs;
}

volSymmTensorField evalTensorAddSub
(
    const wordList& tokens,
    label& i,
    const volScalarField& I1,
    const volScalarField& I2,
    const volSymmTensorField& T1,
    const volSymmTensorField& T2,
    const volSymmTensorField& T3,
    const fvMesh& mesh
)
{
    volSymmTensorField lhs = evalTensorMulDiv(tokens, i, I1, I2, T1, T2, T3, mesh);

    while (i < tokens.size())
    {
        const word op = tokens[i];
        if (op != "+" && op != "-")
        {
            break;
        }
        ++i;

        if (op == "+") lhs += evalTensorMulDiv(tokens, i, I1, I2, T1, T2, T3, mesh);
        else           lhs -= evalTensorMulDiv(tokens, i, I1, I2, T1, T2, T3, mesh);
    }

    return lhs;
}

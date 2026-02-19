/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "kOmegaSSTA.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"
#include "fvc.H"
#include "fvm.H"
#include "volFields.H"
#include "coordinateSystem.H"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{
// * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //
#include "correctionModel.H"
template<class BasicTurbulenceModel>
tmp<volScalarField>
kOmegaSSTA<BasicTurbulenceModel>::kOmegaSSTA::F1
(
    const volScalarField& CDkOmega
) const
{
    tmp<volScalarField> CDkOmegaPlus = max
    (
        CDkOmega,
        dimensionedScalar(dimless/sqr(dimTime), 1.0e-10)
    );

    tmp<volScalarField> arg1 = min
    (
        min
        (
            max
            (
                (scalar(1)/betaStar_)*sqrt(k_)/(omega_*y_),
                scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
            ),
            (4*alphaOmega2_)*k_/(CDkOmegaPlus*sqr(y_))
        ),
        scalar(10)
    );

    return tanh(pow4(arg1));
}

template<class BasicTurbulenceModel>
tmp<volScalarField>
kOmegaSSTA<BasicTurbulenceModel>::kOmegaSSTA::F2() const
{
    tmp<volScalarField> arg2 = min
    (
        max
        (
            (scalar(2)/betaStar_)*sqrt(k_)/(omega_*y_),
            scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
        ),
        scalar(100)
    );

    return tanh(sqr(arg2));
}

template<class BasicTurbulenceModel>
tmp<volScalarField>
kOmegaSSTA<BasicTurbulenceModel>::kOmegaSSTA::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*(this->mu()/this->rho_)/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}

template<class BasicTurbulenceModel>
tmp<volScalarField>
kOmegaSSTA<BasicTurbulenceModel>::kOmegaSSTA::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}


template<class BasicTurbulenceModel>
void kOmegaSSTA<BasicTurbulenceModel>::correctNut
(
    const volScalarField& S2,
    const volScalarField& F2
)
{
    this->nut_ = a1_*k_/max(a1_*omega_, b1_*F2*sqrt(S2));
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void kOmegaSSTA<BasicTurbulenceModel>::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))), F23());
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> kOmegaSSTA<BasicTurbulenceModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    return min(G, (c1_*betaStar_)*this->k_()*this->omega_());
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> kOmegaSSTA<BasicTurbulenceModel>::epsilonByk
(
    const volScalarField::Internal& F1,
    const volScalarField::Internal& F2
) const
{
    return betaStar_*omega_();
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> kOmegaSSTA<BasicTurbulenceModel>::kSource() const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            k_,
            dimVolume*this->rho_.dimensions()*k_.dimensions()/dimTime
        )
    );
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> kOmegaSSTA<BasicTurbulenceModel>::omegaSource() const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            omega_,
            dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
        )
    );
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> kOmegaSSTA<BasicTurbulenceModel>::Qsas
(
    const volScalarField::Internal& S2,
    const volScalarField::Internal& gamma,
    const volScalarField::Internal& beta
) const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            omega_,
            dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
        )
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
kOmegaSSTA<BasicTurbulenceModel>::kOmegaSSTA
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    alphaK1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK1",
            this->coeffDict_,
            0.85
        )
    ),
    alphaK2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK2",
            this->coeffDict_,
            1.0
        )
    ),
    alphaOmega1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega1",
            this->coeffDict_,
            0.5
        )
    ),
    alphaOmega2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega2",
            this->coeffDict_,
            0.856
        )
    ),
    gamma1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma1",
            this->coeffDict_,
            5.0/9.0
        )
    ),
    gamma2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma2",
            this->coeffDict_,
            0.44
        )
    ),
    beta1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta1",
            this->coeffDict_,
            0.075
        )
    ),
    beta2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta2",
            this->coeffDict_,
            0.0828
        )
    ),
    betaStar_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    a1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "a1",
            this->coeffDict_,
            0.31
        )
    ),
    b1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "b1",
            this->coeffDict_,
            1.0
        )
    ),
    c1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "c1",
            this->coeffDict_,
            10.0
        )
    ),
    F3_
    (
        Switch::lookupOrAddToDict
        (
            "F3",
            this->coeffDict_,
            false
        )
    ),

	y_(wallDist::New(this->mesh_).y()),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
/********************************************************************************************/
//define the added part by Yuan
     nonlinearStress_
    (
        IOobject
        (
            IOobject::groupName("nonlinearStress", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_
        ),
        this->mesh_,
        dimensionedSymmTensor
        (
            "nonlinearStress",
            sqr(dimVelocity),
            Zero
        )
    ),
     Rij_
    (
        IOobject
        (
            IOobject::groupName("Rij", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_
        ),
        this->mesh_,
        dimensionedSymmTensor
        (
            "Rij",
            sqr(dimVelocity),
            Zero
        )
    ),
     Rall_
    (
        IOobject
        (
            IOobject::groupName("Rall", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedSymmTensor
        (
            "Rall",
            sqr(dimVelocity),
            Zero
        )
    )
/********************************************************************************************/
{
    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);
}
// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
//***********************add by yuan****************************************//
template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
kOmegaSSTA<BasicTurbulenceModel>::R() const
{
    tmp<volSymmTensorField> tR
    (
        eddyViscosity<RASModel<BasicTurbulenceModel>>::R()
    );
    tR.ref() += nonlinearStress_;
    return tR;
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
kOmegaSSTA<BasicTurbulenceModel>::devRhoReff() const
{
    tmp<volSymmTensorField> tdevRhoReff
    (
        eddyViscosity<RASModel<BasicTurbulenceModel>>::devRhoReff()
    );
    tdevRhoReff.ref() += this->rho_*nonlinearStress_;
    return tdevRhoReff;
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
kOmegaSSTA<BasicTurbulenceModel>::divDevRhoReff
(
    volVectorField& U
) const
{
    return
    (
        fvc::div(this->rho_*nonlinearStress_) +
        eddyViscosity<RASModel<BasicTurbulenceModel>>::divDevRhoReff(U)
    );
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
kOmegaSSTA<BasicTurbulenceModel>::divDevRhoReff
(
    const volScalarField& rho,
    volVectorField& U
) const
{
    return
    (
        fvc::div(rho*nonlinearStress_) +
        eddyViscosity<RASModel<BasicTurbulenceModel>>::divDevRhoReff(rho, U)
    );
}

//**************************************************************************//
template<class BasicTurbulenceModel>
bool kOmegaSSTA<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        alphaK1_.readIfPresent(this->coeffDict());
        alphaK2_.readIfPresent(this->coeffDict());
        alphaOmega1_.readIfPresent(this->coeffDict());
        alphaOmega2_.readIfPresent(this->coeffDict());
        gamma1_.readIfPresent(this->coeffDict());
        gamma2_.readIfPresent(this->coeffDict());
        beta1_.readIfPresent(this->coeffDict());
        beta2_.readIfPresent(this->coeffDict());
        betaStar_.readIfPresent(this->coeffDict());
        a1_.readIfPresent(this->coeffDict());
        b1_.readIfPresent(this->coeffDict());
        c1_.readIfPresent(this->coeffDict());
        F3_.readIfPresent("F3", this->coeffDict());

        return true;
    }
    else
    {
        return false;
    }
}


template<class BasicTurbulenceModel>
void kOmegaSSTA<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    Info<< "Reading field p\n" << endl;
    const volScalarField& p = this->db().objectRegistry::lookupObject<volScalarField>("p");
    volScalarField& nut = this->nut_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))()()
    );

    tmp<volTensorField> tgradU = fvc::grad(U);
    tmp<volVectorField> tgradk = fvc::grad(k_);
    tmp<volVectorField> tgradmagU = fvc::grad(mag(U));
    volScalarField S2(2*magSqr(symm(tgradU())));
    volScalarField S(sqrt(S2));

//******************************add by yuan***********************************//
    // Match Python/SpaRTA feature normalization:
    // tau = 1/omega, S = symm(gradU)/omega, W = skew(gradU)/omega.
    volScalarField tau = 1./max( S/0.31 + this->omegaMin_,(omega_ + this->omegaMin_));
    volScalarField tau2 = sqr(tau);
    volSymmTensorField sij(dev(symm(tgradU())));
    volTensorField omegaij((skew(tgradU())));
    volScalarField I1(tau2*tr(sij & sij));
    volScalarField I2(tau2*tr(omegaij & omegaij));
    volSymmTensorField T1(tau*sij);
    volSymmTensorField T2(tau2*(symm((sij & omegaij) - (omegaij & sij))));
    volSymmTensorField T3(tau2*(symm(sij & sij)) - (scalar(1.0/3.0))*I1*I);

    const bool printSpaRTAScaling =
        this->coeffDict_.found("printSpaRTAScaling")
      ? readBool(this->coeffDict_.lookup("printSpaRTAScaling"))
      : false;

    const bool printSpaRTADiagnostics =
        this->coeffDict_.found("printSpaRTADiagnostics")
      ? readBool(this->coeffDict_.lookup("printSpaRTADiagnostics"))
      : false;

    if (printSpaRTAScaling)
    {
        static bool printedScalingFormula(false);
        if (!printedScalingFormula)
        {
            Info<< "SpaRTA scaling formulas: "
                << "tau=1/max(omega,omegaMin), "
                << "Sij=tau*symm(gradU), Wij=tau*skew(gradU), "
                << "I1=tr(Sij&Sij), I2=tr(Wij&Wij), "
                << "U_b=<U_x>_V, gradU_norm=gradU/U_b."
                << nl;
            printedScalingFormula = true;
        }

        volScalarField Ux(U.component(vector::X));
        volScalarField Uy(U.component(vector::Y));
        volScalarField Uz(U.component(vector::Z));
        volScalarField gradUMag(mag(tgradU()));
        volScalarField T1T1(T1 && T1);
        volScalarField T2T2(T2 && T2);
        volScalarField T3T3(T3 && T3);
        volScalarField T1GradU(T1 && tgradU());
        volScalarField T2GradU(T2 && tgradU());
        volScalarField T3GradU(T3 && tgradU());

        auto printScalingStats =
            [](const char* name, const volScalarField& field)
            {
                Info<< "SpaRTA scaling: name=" << name
                    << " min=" << gMin(field)
                    << " max=" << gMax(field)
                    << " avg=" << gAverage(field)
                    << nl;
            };

        auto printScalingConstant =
            [](const char* name, const scalar value)
            {
                Info<< "SpaRTA scaling: name=" << name
                    << " min=" << value
                    << " max=" << value
                    << " avg=" << value
                    << nl;
            };

        const scalar totalVolume = gSum(this->mesh_.V());
        const scalar ub =
            totalVolume > VSMALL
          ? fvc::domainIntegrate(Ux).value()/totalVolume
          : 0.0;
        const scalar ubAbs = mag(ub);
        const bool hasExpectedUb = this->coeffDict_.found("expectedUb");
        const scalar expectedUb =
            hasExpectedUb ? readScalar(this->coeffDict_.lookup("expectedUb")) : 0.0;
        const scalar ubRelativeWarning =
            this->coeffDict_.found("ubRelativeWarning")
          ? readScalar(this->coeffDict_.lookup("ubRelativeWarning"))
          : 0.05;
        const scalar ubRelativeDiff =
            hasExpectedUb
          ? ubAbs/max(mag(expectedUb), SMALL)
          : 0.0;

        printScalingConstant("U_b", ub);
        printScalingStats("Ux", Ux);
        printScalingStats("Uy", Uy);
        printScalingStats("Uz", Uz);
        printScalingStats("gradU_mag", gradUMag);
        printScalingStats("omega", omega_);
        printScalingStats("S", S);
        printScalingStats("tau", tau);
        printScalingStats("I1", I1);
        printScalingStats("I2", I2);
        printScalingStats("T1:T1", T1T1);
        printScalingStats("T2:T2", T2T2);
        printScalingStats("T3:T3", T3T3);
        printScalingStats("T1:gradU", T1GradU);
        printScalingStats("T2:gradU", T2GradU);
        printScalingStats("T3:gradU", T3GradU);

        if (hasExpectedUb)
        {
            const scalar ubError =
                mag(ub - expectedUb)/max(mag(expectedUb), SMALL);
            printScalingConstant("U_b_expected", expectedUb);
            printScalingConstant("U_b_relative_to_expected", ubRelativeDiff);
            printScalingConstant("U_b_relative_error", ubError);

            if (ubError > ubRelativeWarning)
            {
                WarningInFunction
                    << "SpaRTA scaling warning: runtime U_b=" << ub
                    << " differs from expectedUb=" << expectedUb
                    << " by relative error " << ubError
                    << " (threshold ubRelativeWarning=" << ubRelativeWarning << ")."
                    << nl;
            }
        }

        if (ubAbs <= VSMALL)
        {
            WarningInFunction
                << "SpaRTA scaling warning: |U_b|=" << ubAbs
                << " is too small. Skipping U/U_b and gradU/U_b diagnostics."
                << nl;
        }
        else
        {
            const dimensionedScalar UbScale("UbScale", U.dimensions(), ub);
            const dimensionedScalar UbAbsScale("UbAbsScale", U.dimensions(), ubAbs);

            volVectorField UNorm(U/UbScale);
            volTensorField gradUNorm(tgradU()/UbScale);
            volSymmTensorField sijNorm(symm(gradUNorm));
            volTensorField omegaijNorm(skew(gradUNorm));
            volScalarField SNorm(sqrt(2*magSqr(sijNorm)));
            volScalarField I1Norm(tr(sijNorm & sijNorm));
            volScalarField I2Norm(tr(omegaijNorm & omegaijNorm));
            volSymmTensorField T1Norm(sijNorm);
            volSymmTensorField T2Norm(symm((sijNorm & omegaijNorm) - (omegaijNorm & sijNorm)));
            volSymmTensorField T3Norm(
                (symm(sijNorm & sijNorm)) - (scalar(1.0/3.0))*I1Norm*I
            );

            volScalarField UxOverUb(UNorm.component(vector::X));
            volScalarField UyOverUb(UNorm.component(vector::Y));
            volScalarField UzOverUb(UNorm.component(vector::Z));
            volScalarField gradUMagOverUb(gradUMag/UbAbsScale);
            volScalarField T1NormT1Norm(T1Norm && T1Norm);
            volScalarField T2NormT2Norm(T2Norm && T2Norm);
            volScalarField T3NormT3Norm(T3Norm && T3Norm);
            volScalarField T1NormGradUNorm(T1Norm && gradUNorm);
            volScalarField T2NormGradUNorm(T2Norm && gradUNorm);
            volScalarField T3NormGradUNorm(T3Norm && gradUNorm);

            printScalingStats("Ux/U_b", UxOverUb);
            printScalingStats("Uy/U_b", UyOverUb);
            printScalingStats("Uz/U_b", UzOverUb);
            printScalingStats("gradU_mag/U_b", gradUMagOverUb);
            printScalingStats("S_norm", SNorm);
            printScalingStats("I1_norm", I1Norm);
            printScalingStats("I2_norm", I2Norm);
            printScalingStats("T1_norm:T1_norm", T1NormT1Norm);
            printScalingStats("T2_norm:T2_norm", T2NormT2Norm);
            printScalingStats("T3_norm:T3_norm", T3NormT3Norm);
            printScalingStats("T1_norm:gradU_norm", T1NormGradUNorm);
            printScalingStats("T2_norm:gradU_norm", T2NormGradUNorm);
            printScalingStats("T3_norm:gradU_norm", T3NormGradUNorm);
        }
    }

    if (printSpaRTADiagnostics)
    {
        volScalarField T1T1(T1 && T1);
        volScalarField T2T2(T2 && T2);
        volScalarField T3T3(T3 && T3);
        Info<< "SpaRTA feature stats:"
            << " I1[min,max]=[" << gMin(I1) << ", " << gMax(I1) << "]"
            << " I2[min,max]=[" << gMin(I2) << ", " << gMax(I2) << "]"
            << " T1:T1[min,max]=[" << gMin(T1T1) << ", " << gMax(T1T1) << "]"
            << " T2:T2[min,max]=[" << gMin(T2T2) << ", " << gMax(T2T2) << "]"
            << " T3:T3[min,max]=[" << gMin(T3T3) << ", " << gMax(T3T3) << "]"
            << nl;
    }

    tmp<volVectorField> tgradP = fvc::grad(p);

    const bool hasBModelExpression = this->coeffDict_.found("bModelExpression");
    const bool hasLegacyCorrection = this->coeffDict_.found("correctionModel");
    const bool hasRModelExpression = this->coeffDict_.found("RModelExpression");
    const auto isNumericZeroExpression = [](const string& expr) -> bool
    {
        std::string normalized(expr.c_str());
        normalized.erase
        (
            std::remove_if
            (
                normalized.begin(),
                normalized.end(),
                [](const unsigned char c) { return std::isspace(c) != 0; }
            ),
            normalized.end()
        );

        if (normalized.empty())
        {
            return false;
        }

        if
        (
            normalized.size() >= 2
         && (
                (normalized.front() == '"' && normalized.back() == '"')
             || (normalized.front() == '\'' && normalized.back() == '\'')
            )
        )
        {
            normalized = normalized.substr(1, normalized.size() - 2);
        }

        if (normalized.empty())
        {
            return false;
        }

        try
        {
            std::size_t pos = 0;
            const double value = std::stod(normalized, &pos);
            return pos == normalized.size() && std::abs(value) <= SMALL;
        }
        catch (...)
        {
            return false;
        }
    };

    if (hasBModelExpression && hasLegacyCorrection)
    {
        WarningInFunction
            << "Both bModelExpression and correctionModel found. "
            << "Using bModelExpression and ignoring correctionModel." << nl;
    }

    if (hasBModelExpression || hasLegacyCorrection)
    {
        // CORRECTION MODEL INJECTION (nonlinear stress / b-delta model)
        string rawBExpr =
            hasBModelExpression
          ? string(this->coeffDict_.lookup("bModelExpression"))
          : string(this->coeffDict_.lookup("correctionModel"));
        if (isNumericZeroExpression(rawBExpr))
        {
            nonlinearStress_ = dimensionedSymmTensor
            (
                "nonlinearStress",
                sqr(dimVelocity),
                symmTensor::zero
            );
        }
        else
        {
            // NOTE: Do NOT touch '-' globally (would break 1e-07).
            rawBExpr.replaceAll("(", " ( ");
            rawBExpr.replaceAll(")", " ) ");
            rawBExpr.replaceAll("*", " * ");
            rawBExpr.replaceAll("/", " / ");
            rawBExpr.replaceAll("+", " + ");
            rawBExpr.replaceAll("^", " ^ ");
            rawBExpr.replaceAll(") (", ") (");

            DynamicList<word> bTokDyn;
            {
                std::string s(rawBExpr.c_str());
                std::istringstream iss(s);
                std::string item;
                while (iss >> item)
                {
                    bTokDyn.append(word(item));
                }
            }

            wordList bTokens(bTokDyn.shrink());
            if (debug)
            {
                Info<< "bModelExpression tokens: " << bTokens << nl;
            }

            label bIdx = 0;
            volSymmTensorField result =
                evalTensorAddSub(bTokens, bIdx, I1, I2, T1, T2, T3, this->mesh());
            nonlinearStress_ = 2.0 * k_ * result;
        }
    }
    else
    {
        nonlinearStress_ = dimensionedSymmTensor
        (
            "nonlinearStress",
            sqr(dimVelocity),
            symmTensor::zero
        );
    }

    nonlinearStress_.correctBoundaryConditions();

    Rij_.correctBoundaryConditions();
    volScalarField Rterm(Rij_ && symm(tgradU()));

    if (hasRModelExpression)
    {
        // R correction expression (k-equation residual correction)
        string rawRExpr = string(this->coeffDict_.lookup("RModelExpression"));
        if (isNumericZeroExpression(rawRExpr))
        {
            Rterm = dimensionedScalar("Rterm", Rterm.dimensions(), 0.0);
            Rterm.correctBoundaryConditions();
        }
        else
        {
            // Support expressions produced by Python RCandidateLibrary names
            rawRExpr.replaceAll("T1 : (T_ij dU_i/dx_j)", "N1");
            rawRExpr.replaceAll("T2 : (T_ij dU_i/dx_j)", "N2");
            rawRExpr.replaceAll("T3 : (T_ij dU_i/dx_j)", "N3");
            rawRExpr.replaceAll("T1:(T_ij dU_i/dx_j)", "N1");
            rawRExpr.replaceAll("T2:(T_ij dU_i/dx_j)", "N2");
            rawRExpr.replaceAll("T3:(T_ij dU_i/dx_j)", "N3");

            // NOTE: Do NOT touch '-' globally (would break 1e-07).
            rawRExpr.replaceAll("(", " ( ");
            rawRExpr.replaceAll(")", " ) ");
            rawRExpr.replaceAll("*", " * ");
            rawRExpr.replaceAll("/", " / ");
            rawRExpr.replaceAll("+", " + ");
            rawRExpr.replaceAll("^", " ^ ");
            rawRExpr.replaceAll(") (", ") (");

            DynamicList<word> rTokDyn;
            {
                std::string s(rawRExpr.c_str());
                std::istringstream iss(s);
                std::string item;
                while (iss >> item)
                {
                    rTokDyn.append(word(item));
                }
            }

            wordList rTokens(rTokDyn.shrink());
            if (debug)
            {
                Info<< "RModelExpression tokens: " << rTokens << nl;
            }

            // N1/N2/N3 correspond to R-library contractions, scaled by 2k.
            volScalarField N1(2.0*k_*(T1 && symm(tgradU())));
            volScalarField N2(2.0*k_*(T2 && symm(tgradU())));
            volScalarField N3(2.0*k_*(T3 && symm(tgradU())));

            label rIdx = 0;
            Rterm = evalScalarAddSub
            (
                rTokens,
                rIdx,
                I1,
                I2,
                this->mesh(),
                &N1,
                &N2,
                &N3
            );
            Rterm.correctBoundaryConditions();
        }
    }

    Rall_= ((2.0/3.0)*I)*k_ - this->nut_*dev(twoSymm(tgradU()))+ nonlinearStress_;
    Rall_.correctBoundaryConditions();
//***********************************************************************************//
    volScalarField::Internal GbyNu((dev(twoSymm(tgradU()()))) && tgradU()());
    volScalarField::Internal GbyNuaijx(((dev(twoSymm(tgradU()()))) - (nonlinearStress_/nut())) && tgradU()());
    volScalarField::Internal G(this->GName(), nut()*GbyNuaijx);
    tgradU.clear();
    tgradk.clear();
    tgradP.clear();
    tgradmagU.clear();

    // Update omega and G at the wall
    omega_.boundaryFieldRef().updateCoeffs();

    volScalarField CDkOmega
    (
        (2*alphaOmega2_)*(fvc::grad(k_) & fvc::grad(omega_))/omega_
    );

    volScalarField F1(this->F1(CDkOmega));
    volScalarField F23(this->F23());
    dimensionedScalar nutSmall("nutSmall", this->nut_.dimensions(), 1e-20);
    volScalarField::Internal nutSafe(max(this->nut_(), nutSmall));
    volScalarField::Internal PkTerm(this->Pk(G));
    volScalarField::Internal kOmegaProdLimiter
    (
        (c1_/a1_)*betaStar_*omega_()
       *max(a1_*omega_(), b1_*F23()*sqrt(S2()))
    );
    volScalarField::Internal omegaBaseByNut(GbyNuaijx);
    volScalarField::Internal omegaRByNut(Rterm()/nutSafe);
    volScalarField::Internal omegaProdUnclipped(omegaBaseByNut + omegaRByNut);
    volScalarField::Internal PkPlusR(PkTerm + Rterm());
    volScalarField::Internal omegaProdByNut
    (
        min(omegaProdUnclipped, kOmegaProdLimiter)
    );

    if (printSpaRTADiagnostics)
    {
        const scalar pAbsMax = max(mag(gMin(PkTerm)), mag(gMax(PkTerm)));
        const scalar rAbsMax = max(mag(gMin(Rterm)), mag(gMax(Rterm)));
        const scalar prAbsMax = max(mag(gMin(PkPlusR)), mag(gMax(PkPlusR)));
        volScalarField betaStarOmegaK(betaStar_*omega_*k_);

        Info<< "SpaRTA source stats:"
            << " k[min,max]=[" << gMin(k_) << ", " << gMax(k_) << "]"
            << " omega[min,max]=[" << gMin(omega_) << ", " << gMax(omega_) << "]"
            << " nut[min,max]=[" << gMin(this->nut_) << ", " << gMax(this->nut_) << "]"
            << " Pk[min,max]=[" << gMin(PkTerm) << ", " << gMax(PkTerm) << "]"
            << " R[min,max]=[" << gMin(Rterm) << ", " << gMax(Rterm) << "]"
            << " (Pk+R)[min,max]=[" << gMin(PkPlusR) << ", " << gMax(PkPlusR) << "]"
            << " omegaBaseByNut[min,max]=[" << gMin(omegaBaseByNut) << ", "
            << gMax(omegaBaseByNut) << "]"
            << " omegaRByNut[min,max]=[" << gMin(omegaRByNut) << ", "
            << gMax(omegaRByNut) << "]"
            << " omegaProdByNut[min,max]=[" << gMin(omegaProdByNut) << ", "
            << gMax(omegaProdByNut) << "]"
            << " betaStar*omega*k[min,max]=[" << gMin(betaStarOmegaK)
            << ", " << gMax(betaStarOmegaK) << "]"
            << " max|R|/max|Pk|=" << (pAbsMax > VSMALL ? rAbsMax/pAbsMax : 0.0)
            << " max|Pk+R|=" << prAbsMax
            << nl;
    }

    {
        volScalarField::Internal gamma(this->gamma(F1));
        volScalarField::Internal beta(this->beta(F1));

        // Turbulent frequency equation
        tmp<fvScalarMatrix> omegaEqn
        (
            fvm::ddt(alpha, rho, omega_)
          + fvm::div(alphaRhoPhi, omega_)
          - fvm::laplacian(alpha*rho*DomegaEff(F1), omega_)
         ==
            alpha()*rho()*gamma
           *omegaProdByNut
          - fvm::SuSp((2.0/3.0)*alpha()*rho()*gamma*divU, omega_)
          - fvm::Sp(alpha()*rho()*beta*omega_(), omega_)
          - fvm::SuSp
            (
                alpha()*rho()*(F1() - scalar(1))*CDkOmega()/omega_(),
                omega_
            )
          + Qsas(S2(), gamma, beta)
          + omegaSource()
          + fvOptions(alpha, rho, omega_)
        );

        omegaEqn.ref().relax();
        fvOptions.constrain(omegaEqn.ref());
        omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
        solve(omegaEqn);
        fvOptions.correct(omega_);
        bound(omega_, this->omegaMin_);
    }

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(alpha, rho, k_)
      + fvm::div(alphaRhoPhi, k_)
      - fvm::laplacian(alpha*rho*DkEff(F1), k_)
     ==
        alpha()*rho()*PkTerm
      + alpha()*rho()*Rterm
      - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
      - fvm::Sp(alpha()*rho()*epsilonByk(F1, F23), k_)
      + kSource()
      + fvOptions(alpha, rho, k_)
    );

    kEqn.ref().relax();
    fvOptions.constrain(kEqn.ref());
    solve(kEqn);
    fvOptions.correct(k_);
    bound(k_, this->kMin_);
    const volScalarField nut_(this->nut());
    correctNut(S2, F23);

/*

    solve(vorticityTransportEqn);
    fvOptions.correct(vorticity_);
    vorticity_.correctBoundaryConditions();
*/
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam
} // End namespace RASModels
// ************************************************************************* //

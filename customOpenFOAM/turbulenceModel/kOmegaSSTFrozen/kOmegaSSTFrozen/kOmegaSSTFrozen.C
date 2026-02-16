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

#include "kOmegaSSTFrozen.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"
#include "fvc.H"
#include "fvm.H"
#include "volFields.H"

#include <sstream>
#include <string>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{
// * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //
template<class BasicTurbulenceModel>
tmp<volScalarField>
kOmegaSSTFrozen<BasicTurbulenceModel>::F1
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
kOmegaSSTFrozen<BasicTurbulenceModel>::F2() const
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
kOmegaSSTFrozen<BasicTurbulenceModel>::F3() const
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
kOmegaSSTFrozen<BasicTurbulenceModel>::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}


template<class BasicTurbulenceModel>
void kOmegaSSTFrozen<BasicTurbulenceModel>::correctNut
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
void kOmegaSSTFrozen<BasicTurbulenceModel>::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))), F23());
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> kOmegaSSTFrozen<BasicTurbulenceModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    return min(G, (c1_*betaStar_)*this->k_()*this->omega_());
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> kOmegaSSTFrozen<BasicTurbulenceModel>::epsilonByk
(
    const volScalarField::Internal& F1,
    const volScalarField::Internal& F2
) const
{
    return betaStar_*omega_();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
kOmegaSSTFrozen<BasicTurbulenceModel>::kOmegaSSTFrozen
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

    frozenMode_(false),
    writeTargets_(true),

    UfrozenName_("U_frozen"),
    kfrozenName_("k_frozen"),
    tauFrozenName_("tau_frozen"),
    bijFrozenName_("bij_frozen"),

    UfrozenPtr_(nullptr),
    kfrozenPtr_(nullptr),
    tauFrozenPtr_(nullptr),
    bijFrozenPtr_(nullptr),

    R_
    (
        IOobject
        (
            IOobject::groupName("R", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("R", dimensionSet(0, 2, -3, 0, 0, 0, 0), 0.0)
    ),

    bDelta_
    (
        IOobject
        (
            IOobject::groupName("bDelta", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedSymmTensor("bDelta", dimless, Zero)
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
    )
{
    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);

        // --- Frozen-field settings ---
    if (this->coeffDict_.found("frozenFields"))
    {
        const dictionary& fz = this->coeffDict_.subDict("frozenFields");

        frozenMode_   = Switch(fz.lookupOrDefault("enabled", true));
        writeTargets_ = Switch(fz.lookupOrDefault("writeTargets", true));

        UfrozenName_ = word(fz.lookupOrDefault("U", word("U_frozen")));
        kfrozenName_ = word(fz.lookupOrDefault("k", word("k_frozen")));

        // Optional: user can provide tau or bij
        if (fz.found("tau"))
        {
            tauFrozenName_ = word(fz.lookup("tau"));
        }
        if (fz.found("bij"))
        {
            bijFrozenName_ = word(fz.lookup("bij"));
        }

        if (frozenMode_)
        {
            UfrozenPtr_.reset
            (
                new volVectorField
                (
                    IOobject
                    (
                        UfrozenName_,
                        this->runTime_.timeName(),
                        this->mesh_,
                        IOobject::MUST_READ,
                        IOobject::NO_WRITE
                    ),
                    this->mesh_
                )
            );

            kfrozenPtr_.reset
            (
                new volScalarField
                (
                    IOobject
                    (
                        kfrozenName_,
                        this->runTime_.timeName(),
                        this->mesh_,
                        IOobject::MUST_READ,
                        IOobject::NO_WRITE
                    ),
                    this->mesh_
                )
            );

            // Load tau OR bij (tau preferred)
            const bool hasTau = this->mesh_.template foundObject<volSymmTensorField>(tauFrozenName_)
                             || IOobject
                                (
                                    tauFrozenName_,
                                    this->runTime_.timeName(),
                                    this->mesh_,
                                    IOobject::MUST_READ,
                                    IOobject::NO_WRITE,
                                    false
                                ).typeHeaderOk<volSymmTensorField>(true);

            const bool hasBij = this->mesh_.template foundObject<volSymmTensorField>(bijFrozenName_)
                             || IOobject
                                (
                                    bijFrozenName_,
                                    this->runTime_.timeName(),
                                    this->mesh_,
                                    IOobject::MUST_READ,
                                    IOobject::NO_WRITE,
                                    false
                                ).typeHeaderOk<volSymmTensorField>(true);

            if (hasTau)
            {
                tauFrozenPtr_.reset
                (
                    new volSymmTensorField
                    (
                        IOobject
                        (
                            tauFrozenName_,
                            this->runTime_.timeName(),
                            this->mesh_,
                            IOobject::MUST_READ,
                            IOobject::NO_WRITE
                        ),
                        this->mesh_
                    )
                );
            }
            else if (hasBij)
            {
                bijFrozenPtr_.reset
                (
                    new volSymmTensorField
                    (
                        IOobject
                        (
                            bijFrozenName_,
                            this->runTime_.timeName(),
                            this->mesh_,
                            IOobject::MUST_READ,
                            IOobject::NO_WRITE
                        ),
                        this->mesh_
                    )
                );
            }
            else
            {
                FatalErrorInFunction
                    << "Frozen mode enabled but neither 'tau' nor 'bij' field could be read.\n"
                    << "Provide in turbulenceProperties::frozenFields either:\n"
                    << "  tau <name>;  (preferred)\n"
                    << "or\n"
                    << "  bij <name>;\n"
                    << exit(FatalError);
            }
        }
    }

}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
template<class BasicTurbulenceModel>
bool kOmegaSSTFrozen<BasicTurbulenceModel>::read()
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
void kOmegaSSTFrozen<BasicTurbulenceModel>::correct()
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
    volScalarField& nut = this->nut_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    // --- SpaRTA frozen mode: use frozen U and k when enabled ---
    const volVectorField& Uc = (frozenMode_ && UfrozenPtr_.valid()) ? *UfrozenPtr_ : U;
    const volScalarField& kc = (frozenMode_ && kfrozenPtr_.valid()) ? *kfrozenPtr_ : k_;

    if (frozenMode_ && kfrozenPtr_.valid())
    {
        k_ = kc;
        k_.correctBoundaryConditions();
    }

    // Frozen-mode convection flux
    surfaceScalarField phiFrozen
    (
        IOobject
        (
            "phiFrozen",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        fvc::flux(Uc)
    );

    // Use frozen alphaRhoPhi in frozen mode
    surfaceScalarField alphaRhoPhiUse(alphaRhoPhi);
    if (frozenMode_)
    {
        alphaRhoPhiUse = fvc::interpolate(alpha*rho)*phiFrozen;
    }

    const volTensorField gradU(fvc::grad(Uc));
    const volSymmTensorField Sij(dev(symm(gradU)));
    const volScalarField S2Frozen(2*magSqr(symm(gradU)));
    const volScalarField divU
    (
        frozenMode_
      ? fvc::div(Uc)
      : fvc::div(fvc::absolute(this->phi(), U))
    );

    const dimensionedScalar kSmall("kSmall", kc.dimensions(), SMALL);
    const volScalarField kEff
    (
        IOobject("kEff", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        max(kc, kSmall)
    );

    // Needed by omega wall functions during boundary coefficient updates.
    const volScalarField G
    (
        IOobject(this->GName(), this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        nut*S2Frozen
    );

    omega_.boundaryFieldRef().updateCoeffs();

    volScalarField CDkOmega
    (
        IOobject("CDkOmega", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        (2*alphaOmega2_)*(fvc::grad(k_) & fvc::grad(omega_))/omega_
    );

    const volScalarField F1(this->F1(CDkOmega));
    const volScalarField F23(this->F23());
    const volScalarField gammaF
    (
        IOobject("gammaF", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        F1*(gamma1_ - gamma2_) + gamma2_
    );
    const volScalarField betaF
    (
        IOobject("betaF", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        F1*(beta1_ - beta2_) + beta2_
    );

    // SpaRTA targets and data fields
    volSymmTensorField bijData
    (
        IOobject
        (
            "bijData",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedSymmTensor("bijData", dimless, Zero)
    );

    if (frozenMode_ && tauFrozenPtr_.valid())
    {
        bijData = ((*tauFrozenPtr_)/(2.0*kEff)) - (scalar(1.0/3.0))*I;
        bijData.correctBoundaryConditions();
    }
    else if (frozenMode_ && bijFrozenPtr_.valid())
    {
        bijData = bijFrozenPtr_();
        bijData.correctBoundaryConditions();
    }
    else
    {
        bijData = dimensionedSymmTensor("zero", dimless, Zero);
    }

    const volSymmTensorField b0
    (
        IOobject("b0", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        - (nut / kEff) * Sij
    );
    bDelta_ = bijData - b0;
    bDelta_.correctBoundaryConditions();

    volScalarField PkData
    (
        IOobject
        (
            "PkData",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        (-2.0*kc) * (bijData && gradU)
    );

    PkData = min(PkData, 10.0*betaStar_*omega_*kc);
    PkData.correctBoundaryConditions();

    const volScalarField convK
    (
        IOobject("convK", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        fvc::div(phiFrozen, kc)
    );
    const volScalarField diffK
    (
        IOobject("diffK", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        fvc::laplacian(DkEff(F1), kc)
    );
    R_ = convK - diffK - PkData + betaStar_*omega_*kc;
    R_.correctBoundaryConditions();

    volScalarField PomegaSrc
    (
        IOobject("PomegaSrc", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        G
    );
    if (frozenMode_)
    {
        PomegaSrc = PkData + R_;
    }
    const dimensionedScalar nutSmall("nutSmall", nut.dimensions(), SMALL);
    const volScalarField prod1
    (
        IOobject("prod1", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        PomegaSrc/max(nut, nutSmall)
    );
    const volScalarField prod2
    (
        IOobject("prod2", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        (c1_/a1_)*betaStar_*omega_*max(a1_*omega_, b1_*F23*sqrt(S2Frozen))
    );
    const volScalarField omegaProd
    (
        IOobject("omegaProd", this->runTime_.timeName(), this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        min(prod1, prod2)
    );

    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(alpha, rho, omega_)
      + fvm::div(alphaRhoPhiUse, omega_)
      - fvm::laplacian(alpha*rho*DomegaEff(F1), omega_)
     ==
        alpha*rho*gammaF*omegaProd
      - fvm::SuSp((2.0/3.0)*alpha*rho*gammaF*divU, omega_)
      - fvm::Sp(alpha*rho*betaF*omega_, omega_)
      - fvm::SuSp(alpha*rho*(F1 - scalar(1))*CDkOmega/max(omega_, this->omegaMin_), omega_)
      + fvOptions(alpha, rho, omega_)
    );

    omegaEqn.ref().relax();
    fvOptions.constrain(omegaEqn.ref());
    omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
    solve(omegaEqn);
    fvOptions.correct(omega_);
    bound(omega_, this->omegaMin_);

    if (!frozenMode_)
    {
        const volScalarField::Internal divUi
        (
            fvc::div(fvc::absolute(this->phi(), U))()()
        );

        tmp<fvScalarMatrix> kEqn
        (
            fvm::ddt(alpha, rho, k_)
          + fvm::div(alphaRhoPhi, k_)
          - fvm::laplacian(alpha*rho*DkEff(F1), k_)
         ==
            G
          - fvm::SuSp((2.0/3.0)*alpha*rho*divUi, k_)
          - fvm::Sp(alpha*rho*betaStar_*omega_, k_)
          + fvOptions(alpha, rho, k_)
        );

        kEqn.ref().relax();
        fvOptions.constrain(kEqn.ref());
        solve(kEqn);
        fvOptions.correct(k_);
        bound(k_, this->kMin_);
    }
    else
    {
        k_ = kc;
        k_.correctBoundaryConditions();
    }
    correctNut(S2Frozen, F23);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam
} // End namespace RASModels
// ************************************************************************* //

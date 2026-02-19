/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  7
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Compute SpaRTA feature fields from U and omega at selected time directories.

    Written fields:
        gradU, tau, Sij, Wij, I1, I2, T1, T2, T3

    Formulas (matched with kOmegaSSTA runtime implementation):
        S   = sqrt(2*magSqr(symm(gradU)))
        tau = 1 / max(S/0.31 + omegaFloor, omega + omegaFloor)
        Sij = tau * dev(symm(gradU))
        Wij = tau * skew(gradU)
        I1  = tr(Sij & Sij)
        I2  = tr(Wij & Wij)
        T1  = Sij
        T2  = symm((Sij & Wij) - (Wij & Sij))
        T3  = symm(Sij & Sij) - (1/3) * I1 * I

Usage
    spartaFeaturePostProcess -case <caseDir> [-time <range>] [-omegaFloor <scalar>]

    Then sample to your 2D plane with your existing surfaceSampling dictionary:
        postProcess -case <caseDir> -func surfaceSampling -time <same time>
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "timeSelector.H"

int main(int argc, char *argv[])
{
    timeSelector::addOptions();
    argList::addOption
    (
        "omegaFloor",
        "scalar",
        "Offset used in tau = 1/max(S/0.31 + omegaFloor, omega + omegaFloor)."
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    scalar omegaFloor = SMALL;
    if (args.optionFound("omegaFloor"))
    {
        omegaFloor = readScalar(IStringStream(args.optionLookup("omegaFloor"))());
    }

    if (omegaFloor <= 0)
    {
        FatalErrorInFunction
            << "omegaFloor must be > 0. Got " << omegaFloor << nl
            << exit(FatalError);
    }

    const instantList timeDirs = timeSelector::select0(runTime, args);

    Info<< "spartaFeaturePostProcess formulas: "
        << "S=sqrt(2*magSqr(symm(gradU))), "
        << "tau=1/max(S/0.31+omegaFloor,omega+omegaFloor), "
        << "Sij=tau*dev(symm(gradU)), Wij=tau*skew(gradU), "
        << "I1=tr(Sij&Sij), I2=tr(Wij&Wij)."
        << nl;

    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);
        Info<< "\nProcessing time = " << runTime.timeName() << nl << endl;

        IOobject UHeader
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        );

        IOobject omegaHeader
        (
            "omega",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        );

        if
        (
            !UHeader.typeHeaderOk<volVectorField>(true)
         || !omegaHeader.typeHeaderOk<volScalarField>(true)
        )
        {
            WarningInFunction
                << "Skipping time " << runTime.timeName()
                << " because U or omega is missing/incompatible." << nl;
            continue;
        }

        volVectorField U(UHeader, mesh);
        volScalarField omega(omegaHeader, mesh);

        tmp<volTensorField> tgradU = fvc::grad(U);

        volTensorField gradU
        (
            IOobject
            (
                "gradU",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            tgradU()
        );

        const dimensionedScalar omegaFloorDim("omegaFloor", omega.dimensions(), omegaFloor);

        volScalarField S
        (
            IOobject
            (
                "S",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            sqrt(2*magSqr(symm(gradU)))
        );

        volScalarField tau
        (
            IOobject
            (
                "tau",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            1.0/max((S/scalar(0.31) + omegaFloorDim), (omega + omegaFloorDim))
        );

        volSymmTensorField Sij
        (
            IOobject
            (
                "Sij",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            tau*dev(symm(gradU))
        );

        volTensorField Wij
        (
            IOobject
            (
                "Wij",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            tau*skew(gradU)
        );

        volScalarField I1
        (
            IOobject
            (
                "I1",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            tr(Sij & Sij)
        );

        volScalarField I2
        (
            IOobject
            (
                "I2",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            tr(Wij & Wij)
        );

        volSymmTensorField T1
        (
            IOobject
            (
                "T1",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            Sij
        );

        volSymmTensorField T2
        (
            IOobject
            (
                "T2",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            symm((Sij & Wij) - (Wij & Sij))
        );

        const dimensionedSymmTensor Iden("I", dimless, symmTensor::I);

        volSymmTensorField T3
        (
            IOobject
            (
                "T3",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            symm(Sij & Sij) - (scalar(1.0/3.0))*I1*Iden
        );

        gradU.write();
        tau.write();
        Sij.write();
        Wij.write();
        I1.write();
        I2.write();
        T1.write();
        T2.write();
        T3.write();

        Info<< "Wrote SpaRTA fields at time " << runTime.timeName()
            << " with omegaFloor=" << omegaFloor
            << ". I1[min,max]=[" << gMin(I1) << ", " << gMax(I1) << "]"
            << " I2[min,max]=[" << gMin(I2) << ", " << gMax(I2) << "]"
            << nl;
    }

    Info<< "\nEnd\n" << endl;
    return 0;
}


// ************************************************************************* //

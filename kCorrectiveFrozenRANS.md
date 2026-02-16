# kCorrective Frozen RANS User Guide

This guide explains how to run the custom frozen corrective RANS workflow using:

- Solver: `frozenSimpleFoam`
- Model: `kOmegaSSTFrozen`
- Example case: `dataset/test-baseline/frozenAR_1_180`

## 1. Prerequisites

1. OpenFOAM 7 environment is available.
2. Custom sources exist in:
   - `customOpenFOAM/turbulenceModel/kOmegaSSTFrozen`
   - `customOpenFOAM/solvers/frozenSimpleFoam`
3. The case contains frozen fields in `0/`:
   - `U_frozen`
   - `k_frozen`
   - `tau_frozen`

## 2. Build

Always clean first, then build.

```bash
source /opt/openfoam7/etc/bashrc

cd customOpenFOAM/turbulenceModel/kOmegaSSTFrozen
wclean libso
wmake libso

cd ../../solvers/frozenSimpleFoam
wclean
wmake
```

Expected artifacts:

- `~/OpenFOAM/<user>-7/platforms/linux64GccDPInt32Opt/lib/libkOmegaSSTFrozen.so`
- `~/OpenFOAM/<user>-7/platforms/linux64GccDPInt32Opt/bin/frozenSimpleFoam`

## 3. Case Configuration

### 3.1 `system/controlDict`

Use the frozen solver and custom turbulence library:

```foam
libs        ("libkOmegaSSTFrozen.so");
application frozenSimpleFoam;
```

### 3.2 `constant/turbulenceProperties`

Use `kOmegaSSTFrozen` and map frozen fields explicitly:

```foam
simulationType  RAS;

RAS
{
    RASModel        kOmegaSSTFrozen;
    turbulence      on;
    printCoeffs     on;

    kOmegaSSTFrozenCoeffs
    {
        frozenFields
        {
            enabled      true;
            writeTargets true;
            U            U_frozen;
            k            k_frozen;
            tau          tau_frozen;
        }
    }
}
```

### 3.3 `system/fvSchemes`

Add frozen divergence schemes:

```foam
div(U_frozen)          Gauss linear;
div(phiFrozen,k_frozen) bounded Gauss upwind;
div(phiFrozen,k)       bounded Gauss upwind;
```

## 4. Frozen Field Requirements

### 4.1 `0/tau_frozen`

`tau_frozen` must be a valid `volSymmTensorField` OpenFOAM file with a complete `FoamFile` header.

Minimum valid header:

```foam
FoamFile
{
    version     2.0;
    format      ascii;
    class       volSymmTensorField;
    location    "0";
    object      tau_frozen;
}
```

### 4.2 `tau_frozen` dimensions

Use:

```foam
dimensions [0 2 -2 0 0 0 0];
```

This is required because the model computes anisotropy as `tau/(2*k)`.

### 4.3 `tau_frozen` wall patch type

For this setup, `fixedWalls` must use a valid tensor patch type such as:

```foam
type zeroGradient;
```

Do not use `noSlip` for `volSymmTensorField`.

## 5. Run

```bash
source /opt/openfoam7/etc/bashrc
frozenSimpleFoam -case dataset/test-baseline/frozenAR_1_180
```

## 6. Useful Checks

1. Confirm model selection in log:
   - `Selecting RAS turbulence model kOmegaSSTFrozen`
2. Confirm time loop advances:
   - multiple `Time = ...` lines without `FOAM FATAL`.

## 7. Common Errors and Fixes

1. `keyword version is undefined in .../0/tau_frozen`
   - Fix `FoamFile` header in `tau_frozen`.

2. `keyword div(U_frozen) is undefined in .../fvSchemes.divSchemes`
   - Add frozen div entries in `system/fvSchemes`.

3. Dimension mismatch around `tau/(2*k)`
   - Set `tau_frozen` dimensions to `[0 2 -2 0 0 0 0]`.

4. Wrong turbulence model selected
   - Ensure `RASModel kOmegaSSTFrozen;` in `constant/turbulenceProperties`.

## 8. Notes

1. The model can also use `bij` input instead of `tau` if configured.
2. `writeTargets true` writes model target fields (for diagnostics/training workflows).

/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2012 OpenFOAM Foundation
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

Contributors/Copyright
    2014 Hagen Müller <hagen.mueller@unibw.de> Universität der Bundeswehr München
    2014 Likun Ma <L.Ma@tudelft.nl> TU Delft
    2019 Lim Wei Xian <weixian001@e.ntu.edu.sg> NTUsg

\*---------------------------------------------------------------------------*/

#include "heRhoThermo.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
void Foam::heRhoThermo<BasicPsiThermo, MixtureType>::calculate()
{
    const scalarField& hCells = this->he().internalField();
    const scalarField& pCells = this->p_.internalField();
    /*const scalarField& ZCells = Z_.internalField();
    const scalarField& varZCells = varZ_.internalField();
    const scalarField& chiCells = Chi_.internalField();*/
    
    // bound the progress variable
   /* scalar chiMax = solver_.maxChi();
    scalarList x(3, 0.0);
    double Zeta;*/

    scalarField& TCells = this->T_.internalField();
    scalarField& psiCells = this->psi_.internalField();
    scalarField& rhoCells = this->rho_.internalField();
    scalarField& muCells = this->mu_.internalField();
    scalarField& alphaCells = this->alpha_.internalField();

    forAll(TCells, celli)
    {
        /*Zeta = sqrt(varZCells[celli]/max(ZCells[celli]*(1 - ZCells[celli]), SMALL));
        x[0] = min(chiMax, max(0,chiCells[celli]));
        x[1] = min(Zeta, 0.99);
        x[2] = ZCells[celli];	

        ubIF_[celli] = solver_.upperBounds(x);
        posIF_[celli] = solver_.position(ubIF_[celli], x);
        
        TCells[celli] = solver_.interpolate(ubIF_[celli], posIF_[celli], 0);
        psiCells[celli] = solver_.interpolate(ubIF_[celli], posIF_[celli], 1);
        rhoCells[celli] = solver_.interpolate(ubIF_[celli], posIF_[celli], 2);
        muCells[celli] = solver_.interpolate(ubIF_[celli], posIF_[celli], 3);*/
        const typename MixtureType::thermoType& mixture_ =
            this->cellMixture(celli);

        TCells[celli] = mixture_.THE(hCells[celli],pCells[celli],TCells[celli]);
        psiCells[celli] = mixture_.psi(pCells[celli], TCells[celli]);
        rhoCells[celli] = mixture_.rho(pCells[celli], TCells[celli]);
        muCells[celli] = mixture_.mu(pCells[celli], TCells[celli]);
        alphaCells[celli] = mixture_.alphah(pCells[celli], TCells[celli]);
    }

/*    forAll(this->T_.boundaryField(), patchi)
    {
        const fvPatchScalarField& pChi = Chi_.boundaryField()[patchi];
        const fvPatchScalarField& pvarZ = varZ_.boundaryField()[patchi];
        const fvPatchScalarField& pZ = Z_.boundaryField()[patchi];
        
        fvPatchScalarField& pT = this->T_.boundaryField()[patchi];
        fvPatchScalarField& ppsi = this->psi_.boundaryField()[patchi];
        fvPatchScalarField& prho = this->rho_.boundaryField()[patchi];
        fvPatchScalarField& pmu = this->mu_.boundaryField()[patchi];

        forAll(pT, facei)
        {
            Zeta = sqrt(pvarZ[facei]/max(pZ[facei]*(1 - pZ[facei]), SMALL));
            x[0] = min(chiMax, max(0, pChi[facei]));
            x[1] = min(Zeta, 0.99);
            x[2] = pZ[facei];
            
            ubP_[facei] = solver_.upperBounds(x);
            posP_[facei] = solver_.position(ubP_[facei], x);

            pT[facei] = solver_.interpolate(ubP_[facei], posP_[facei], 0);
            ppsi[facei] = solver_.interpolate(ubP_[facei], posP_[facei], 1);
            prho[facei] = solver_.interpolate(ubP_[facei], posP_[facei], 2);
            pmu[facei] = solver_.interpolate(ubP_[facei], posP_[facei], 3);
         }
    }*/
    
    forAll(this->T_.boundaryField(), patchi)
    {
        fvPatchScalarField& pp = this->p_.boundaryField()[patchi];
        fvPatchScalarField& pT = this->T_.boundaryField()[patchi];
        fvPatchScalarField& ppsi = this->psi_.boundaryField()[patchi];
        fvPatchScalarField& prho = this->rho_.boundaryField()[patchi];

        fvPatchScalarField& ph = this->he().boundaryField()[patchi];

        fvPatchScalarField& pmu = this->mu_.boundaryField()[patchi];
        fvPatchScalarField& palpha = this->alpha_.boundaryField()[patchi];

        forAll(pT, facei)
        {
            const typename MixtureType::thermoType& mixture_ =
                 this->patchFaceMixture(patchi, facei);

            pT[facei] = mixture_.THE(ph[facei], pp[facei], pT[facei]);

            ppsi[facei] = mixture_.psi(pp[facei], pT[facei]);
            prho[facei] = mixture_.rho(pp[facei], pT[facei]);
            pmu[facei] = mixture_.mu(pp[facei], pT[facei]);
            palpha[facei] = mixture_.alphah(pp[facei], pT[facei]);
         }
    }
    
    Info << "----------> T min/max =" << min(this->T_).value() << ", "
        << max(this->T_).value() << endl;
    Info << "----------> psi min/max =" << min(this->psi_).value() << ", "
        << max(this->psi_).value() << endl;
    Info << "----------> rho min/max =" << min(this->rho_).value() << ", "
        << max(this->rho_).value() << endl;
    Info << "----------> mu min/max =" << min(this->mu_).value() << ", "
        << max(this->mu_).value() << endl;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
Foam::heRhoThermo<BasicPsiThermo, MixtureType>::heRhoThermo
(
    const fvMesh& mesh,
    const word& phaseName
)
:
    heThermo<BasicPsiThermo, MixtureType>(mesh, phaseName)
 /*   solver_(Foam::combustionModels::tableSolver(mesh, tables())),
    Z_(Foam::fluidThermo::Z()),
    varZ_(Foam::fluidThermo::varZ()),
    Chi_(Foam::fluidThermo::Chi()),
    ubIF_(mesh.cells().size()),
    ubP_(),
    posIF_(mesh.cells().size()),
    posP_()  
{
    const polyBoundaryMesh& patches = mesh.boundaryMesh();
	int patchSize = 0;
    forAll(patches, patchI)
    {
    	const polyPatch& pp = patches[patchI];
    	if (pp.size() > patchSize) patchSize = pp.size();
    }

    ubP_.setSize(patchSize);
    posP_.setSize(patchSize);
} */
{}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
Foam::heRhoThermo<BasicPsiThermo, MixtureType>::~heRhoThermo()
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

/*template<class BasicPsiThermo, class MixtureType>
Foam::hashedWordList Foam::heRhoThermo<BasicPsiThermo, MixtureType>::tables()
{
	hashedWordList tableNames;
	tableNames.append("T");			
	tableNames.append("psi");		
	tableNames.append("rho");		
	tableNames.append("mu");		
	
	return tableNames;
}*/

template<class BasicPsiThermo, class MixtureType>
void Foam::heRhoThermo<BasicPsiThermo, MixtureType>::correct()
{
    if (debug)
    {
        Info<< "entering heRhoThermo<MixtureType>::correct()" << endl;
    }

    calculate();

    if (debug)
    {
        Info<< "exiting heRhoThermo<MixtureType>::correct()" << endl;
    }
}


// ************************************************************************* //

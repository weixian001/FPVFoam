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

#include "RASModel.H"
#include "wallFvPatch.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace compressible
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(RASModel, 0);
defineRunTimeSelectionTable(RASModel, dictionary);
addToRunTimeSelectionTable(turbulenceModel, RASModel, turbulenceModel);

// * * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * //

void RASModel::printCoeffs()
{
    if (printCoeffs_)
    {
        Info<< type() << "Coeffs" << coeffDict_ << endl;
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

RASModel::RASModel
(
    const word& type,
    const volScalarField& rho,
    const volVectorField& U,
    const surfaceScalarField& phi,
    fluidThermo& thermophysicalModel,              
    const word& turbulenceModelName
)
:
    turbulenceModel(rho, U, phi, thermophysicalModel, turbulenceModelName),

    IOdictionary
    (
        IOobject
        (
            "RASProperties",
            U.time().constant(),
            U.db(),
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    ),

    turbulence_(lookup("turbulence")),
    printCoeffs_(lookupOrDefault<Switch>("printCoeffs", false)),
    reactingCondition_(lookupOrDefault<Switch>("reactingCondition", true)),	//added
    coeffDict_(subOrEmptyDict(type + "Coeffs")),
    Cchi_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cchi",
            coeffDict_,
            2.0
        )
    ),
    Sc_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Sc",
            coeffDict_,
            1.0
        )
    ),
    Sct_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Sct",
            coeffDict_,
            0.9
        )
    ),
    kMin_("kMin", sqr(dimVelocity), SMALL),
    epsilonMin_("epsilonMin", kMin_.dimensions()/dimTime, SMALL),
    omegaMin_("omegaMin", dimless/dimTime, SMALL),
    varZ_(thermophysicalModel.varZ()),
    Chi_(thermophysicalModel.Chi()),
    Srr_(thermophysicalModel.Srr())		//added
{
    kMin_.readIfPresent(*this);
    epsilonMin_.readIfPresent(*this);
    omegaMin_.readIfPresent(*this);

    // Force the construction of the mesh deltaCoeffs which may be needed
    // for the construction of the derived models and BCs
    mesh_.deltaCoeffs();
}


// * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * * //

autoPtr<RASModel> RASModel::New
(
    const volScalarField& rho,
    const volVectorField& U,
    const surfaceScalarField& phi,
    fluidThermo& thermophysicalModel,         
    const word& turbulenceModelName
)
{
    // get model name, but do not register the dictionary
    // otherwise it is registered in the database twice
    const word modelType
    (
        IOdictionary
        (
            IOobject
            (
                "RASProperties",
                U.time().constant(),
                U.db(),
                IOobject::MUST_READ_IF_MODIFIED,
                IOobject::NO_WRITE,
                false
            )
        ).lookup("RASModel")
    );

    Info<< "Selecting RAS turbulence model " << modelType << endl;

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn
        (
            "RASModel::New"
            "("
                "const volScalarField&, "
                "const volVectorField&, "
                "const surfaceScalarField&, "
                "fluidThermo&, "
                "const word&"
            ")"
        )   << "Unknown RASModel type "
            << modelType << nl << nl
            << "Valid RASModel types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<RASModel>
    (
        cstrIter()(rho, U, phi, thermophysicalModel, turbulenceModelName)
    );
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void RASModel::correct()
{
    turbulenceModel::correct();
}


void RASModel::correctVarZ()
{
	tmp<fvScalarMatrix> varZEqn
    (
        (
          	  fvm::ddt(rho_, varZ_)
            + fvm::div(phi_, varZ_)
            - fvm::laplacian(DZEff(), varZ_)
            - 2.0*DZEff()*magSqr(fvc::grad(this->thermo().Z()))
            + Cchi_*rho_*epsilon()/k()*varZ_
        )
    );

    varZEqn().relax();
    varZEqn().boundaryManipulate(varZ_.boundaryField());

    solve(varZEqn);
    varZ_.correctBoundaryConditions();
    bound(varZ_, 0.0);

    //Info<< "----------> varZ min/max   = " << min(varZ_).value() << ", "
      //  << max(varZ_).value() << endl;
}

/*void RASModel::correctChi()
{
    Chi_ = Cchi_ * epsilon()/k() * varZ_;
    bound(Chi_, 0.0);

    Info<< "----------> chi min/max   = " << min(Chi_).value() << ", "
        << max(Chi_).value() << endl;
}*/

void RASModel::correctChi()
{
	if (reactingCondition_)
	{
		tmp<fvScalarMatrix> PvEqn
		(
			(
			  fvm::ddt(rho_, Chi_)
			+ fvm::div(phi_, Chi_)
        		- fvm::laplacian(DZEff(), Chi_)
        		- rho_*Srr_
        		)
    		);
    		PvEqn().relax();
    		PvEqn().boundaryManipulate(Chi_.boundaryField());
    		solve(PvEqn);
    		Chi_.correctBoundaryConditions();
  
    	}
    	else
    	{	
    		tmp<fvScalarMatrix> PvEqn
		(
			(
			  fvm::ddt(rho_, Chi_)
			+ fvm::div(phi_, Chi_)
        		- fvm::laplacian(DZEff(), Chi_)
        		)
    		);
    		PvEqn().relax();
    		PvEqn().boundaryManipulate(Chi_.boundaryField());
    		solve(PvEqn);
    		Chi_.correctBoundaryConditions();
    	}
    	
    	/*PvEqn().relax();
    	PvEqn().boundaryManipulate(Chi_.boundaryField());

    	solve(PvEqn);
    	Chi_.correctBoundaryConditions();
    	//bound(Chi_, 0.0);*/
    	
    	//Info<< "----------> chi min/max   = " << min(Chi_).value() << ", "
        //<< max(Chi_).value() << endl;
}

void RASModel::correctSrr()		//added
{
	Srr_ = this->thermo().Srr();
	bound(Srr_, 0.0);

    //Info<< "----------> Srr min/max   = " << min(Srr_).value() << ", "
      //  << max(Srr_).value() << endl;
}

bool RASModel::read()
{
    //if (regIOobject::read())

    // Bit of trickery : we are both IOdictionary ('RASProperties') and
    // an regIOobject from the turbulenceModel level. Problem is to distinguish
    // between the two - we only want to reread the IOdictionary.

    bool ok = IOdictionary::readData
    (
        IOdictionary::readStream
        (
            IOdictionary::type()
        )
    );
    IOdictionary::close();

    if (ok)
    {
        lookup("turbulence") >> turbulence_;

        if (const dictionary* dictPtr = subDictPtr(type() + "Coeffs"))
        {
            coeffDict_ <<= *dictPtr;
        }

        kMin_.readIfPresent(*this);
        epsilonMin_.readIfPresent(*this);
        omegaMin_.readIfPresent(*this);

        return true;
    }
    else
    {
        return false;
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace compressible
} // End namespace Foam

// ************************************************************************* //

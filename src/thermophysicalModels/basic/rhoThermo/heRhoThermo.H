/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2013 OpenFOAM Foundation
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

Class
    Foam::heRhoThermo

Description
    Energy for a mixture based on density

SourceFiles
    heRhoThermo.C

Contributors/Copyright
    2014 Hagen Müller <hagen.mueller@unibw.de> Universität der Bundeswehr München
    2014 Likun Ma <L.Ma@tudelft.nl> TU Delft
    2019 Lim Wei Xian <weixian001@e.ntu.edu.sg> NTUsg


\*---------------------------------------------------------------------------*/

#ifndef heRhoThermo_H
#define heRhoThermo_H

#include "rhoThermo.H"
#include "heThermo.H"
//#include "tableSolver.H"
//#include "hashedWordList.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class heRhoThermo Declaration
\*---------------------------------------------------------------------------*/

template<class BasicPsiThermo, class MixtureType>
class heRhoThermo
:
    public heThermo<BasicPsiThermo, MixtureType>
{
    protected:

    // Protected data

    /*    // tables
	Foam::combustionModels::tableSolver solver_;
	
	//- Mixture fraction
        const volScalarField& Z_;
        
        //- Mixture fraction variance
        const volScalarField& varZ_;

        //- Progress Variable
        const volScalarField& Chi_;
        
        //- Progress Variable
        //volScalarField& Chi_;
        
        //- Upper Bounds for table interpolation
        List<List<int> > ubIF_, ubP_;

        //- Table Position
        List<scalarList> posIF_, posP_;    */
    
    // Private Member Functions

        //- Calculate the thermo variables
        void calculate();

        //- Construct as copy (not implemented)
        heRhoThermo(const heRhoThermo<BasicPsiThermo, MixtureType>&);


public:

    //- Runtime type information
    TypeName("heRhoThermo");


    // Constructors

        //- Construct from mesh and phase name
        heRhoThermo
        (
            const fvMesh&,
            const word& phaseName
        );


    //- Destructor
    virtual ~heRhoThermo();


    // Member functions

        //hashedWordList tables();
        
        //- Update properties
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
#   include "heRhoThermo.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011 OpenFOAM Foundation
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
    Foam::pureMixture

Description
    Foam::pureMixture

SourceFiles
    pureMixture.C

\*---------------------------------------------------------------------------*/

#ifndef pureMixture_H
#define pureMixture_H

#include "basicMixture.H"
//#include "tableSolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class pureMixture Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType>
class pureMixture
:
    public basicMixture
{
    // Private data

        ThermoType mixture_;

        //- Construct as copy (not implemented)
        pureMixture(const pureMixture<ThermoType>&);
        
        // tables
	//Foam::combustionModels::tableSolver solver_;


public:

    //- The type of thermodynamics this mixture is instantiated for
    typedef ThermoType thermoType;


    // Constructors

        //- Construct from dictionary and mesh
        pureMixture(const dictionary&, const fvMesh&);


    // Member functions

        const ThermoType& cellMixture(const label) const
        {
            return mixture_;
        }

        const ThermoType& patchFaceMixture
        (
            const label,
            const label
        ) const
        {
            return mixture_;
        }

        const ThermoType& cellVolMixture
        (
            const scalar,
            const scalar,
            const label
        ) const
        {
            return mixture_;
        }

        const ThermoType& patchFaceVolMixture
        (
            const scalar,
            const scalar,
            const label,
            const label
        ) const
        {
            return mixture_;
        }
        
        //hashedWordList tables();

        //- Read dictionary
        void read(const dictionary&);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#ifdef NoRepository
#   include "pureMixture.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

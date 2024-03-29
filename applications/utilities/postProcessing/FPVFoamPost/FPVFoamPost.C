/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 1991-2009 OpenCFD Ltd.
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM; if not, write to the Free Software Foundation,
    Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

Description
    Calculates species mass fractions and thermodynamic properties
    from given Z, chi and Zeta fields

Contributors/Copyright
    2014 Hagen Müller <hagen.mueller@unibw.de> Universität der Bundeswehr München
    2019 Lim Wei Xian <weixian001@e.ntu.edu.sg> NTUsg
    

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "timeSelector.H"
#include "tableSolver.H"

#include "fvCFD.H"
#include "rhoCombustionModel.H"
#include "IOdictionary.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    timeSelector::addOptions();
    #include "addRegionOption.H"
    argList::addOption
    (
        "fields",
        "list",
        "specify a list of fields to be reconstructed. Eg, '(U T p)' - "
        "regular expressions not currently supported"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    instantList timeDirs = timeSelector::select0(runTime, args);

    HashSet<word> selectedFields;
    if (args.optionFound("fields"))
    {
        args.optionLookup("fields")() >> selectedFields;
    }

    Info<< "Creating combustion model\n" << endl;

    autoPtr<combustionModels::rhoCombustionModel> combustion
    (
        combustionModels::rhoCombustionModel::New
        (
            mesh
        )
    );

    rhoReactionThermo& thermo = combustion->thermo();    

    const IOdictionary combProps
    (
        IOobject
        (
            "combustionProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const IOdictionary tableProps
    (
        IOobject
        (
            "tableProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const word combTypeName = combProps.lookup("combustionModel");
    const label tempOpen = combTypeName.find('<');
    const word modelType = combTypeName(0, tempOpen);
    dictionary coeffs_(combProps.subDict(modelType + "Coeffs"));

    //Switch useScalarDissipation_(coeffs_.lookup("useScalarDissipation"));
    Switch useProgressVariable_(coeffs_.lookup("useProgressVariable"));			//added
    Switch useMixtureFractionVariance_(coeffs_.lookup("useMixtureFractionVariance"));

    scalarList Pv_Param(tableProps.lookup("Pv_param"));		//amended
    //scalar CLimiter = max(C_Param);

    scalarList x(3);
    scalarList y(3);
    List<List<int> > ubIF(mesh.cells().size());
    List<scalarList> posIF(mesh.cells().size());
    List<List<int> > ubP(mesh.faces().size());
    List<scalarList> posP(mesh.faces().size());

    wordList tableNames(thermo.composition().species());
    tableNames.append("he");
    tableNames.append("Srr");
    word LambdaNames = "chi_lambda_table";
    Foam::combustionModels::tableSolver solver(Foam::combustionModels::tableSolver(mesh, tableNames, LambdaNames));

    PtrList<volScalarField>& Y(thermo.composition().Y());
    volScalarField& he(thermo.he());
    volScalarField& Srr(thermo.Srr());

    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);
        Info << nl << "Time = " << runTime.timeName() << nl << endl;

        volScalarField Z
        (
            IOobject
            (
                "Zmix",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        
        /*volScalarField Pv
        (
            IOobject
            (
                "Pv",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );*/
        
        volScalarField chi
        (
            IOobject
            (
                "chi",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        volScalarField varZ
        (
            IOobject
            (
                "varZ",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );

        volScalarField Zeta
        (
            IOobject
            (
                "Zeta",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            sqrt(varZ/max(Z*(1-Z), SMALL))
        );

        // Interpolate for internal Field
//        scalar LambdaMax = solver.maxChi();
        double Lambda;
//	scalar chiMin = solver.minChi();
        forAll(Y, i)
        {
     	  scalarField& YCells = Y[i].internalField();

           forAll(Z, cellI)
           {
         	 if (i == 0)
         	 {

                 y[0] = min(Zeta[cellI], 0.99);
                 y[1] = Z[cellI];
                 y[2] = max(0.0, chi[cellI]);
                 //y[2] = chi[cellI];

                 List<int> ubCL_ = solver.upperBounds_Lambda(y);
                 scalarList posCL_ = solver.position_Lambda(ubCL_, y);

                 scalarList Lambda_ = solver.interpolateS(ubCL_, posCL_);
                 Lambda = solver.upperBoundsLambda_(y[2], Lambda_);

                  //if (useScalarDissipation_)   x[0] = min(chi[cellI], chiLimiter);
                  //if (useProgressVariable_)   x[0] = min(Lambda, LambdaMax);      //amended
                  if (useProgressVariable_)   x[0] = Lambda;      //amended
                  if (useMixtureFractionVariance_) x[1] = min(Zeta[cellI], 0.99);
                  x[2] = Z[cellI];

                  ubIF[cellI] = solver.upperBounds(x);
                  posIF[cellI] = solver.position(ubIF[cellI], x);

             	 he[cellI] = solver.interpolate(ubIF[cellI], posIF[cellI], (solver.sizeTableNames() - 2));
             	 Srr[cellI] = solver.interpolate(ubIF[cellI], posIF[cellI], (solver.sizeTableNames() - 1));
         	 }

         	 YCells[cellI] = solver.interpolate(ubIF[cellI], posIF[cellI], i);
           }
        }

        // Interpolate for patches
        forAll(he.boundaryField(), patchi)
        {
           const fvPatchScalarField& pChi = chi.boundaryField()[patchi];
           //const fvPatchScalarField& pPv = pPv.boundaryField()[patchi];		//added
           const fvPatchScalarField& pZeta = Zeta.boundaryField()[patchi];
           const fvPatchScalarField& pZ = Z.boundaryField()[patchi];

           fvPatchScalarField& pHe = he.boundaryField()[patchi];
           fvPatchScalarField& pSrr = Srr.boundaryField()[patchi];		//added

           forAll(Y, i)
           {
         	  fvPatchScalarField& pY = Y[i].boundaryField()[patchi];

               forAll(pY , facei)
               {
              	 if (i == 0)
              	 {

                     y[0] = min(Zeta[facei], 0.99);
                     y[1] = pZ[facei];
                     y[2] = min(0.0, pChi[facei]);

                     List<int> ubCL_ = solver.upperBounds_Lambda(y);
                     scalarList posCL_ = solver.position_Lambda(ubCL_, y);

                     scalarList Lambda_ = solver.interpolateS(ubCL_, posCL_);
                     Lambda = solver.upperBoundsLambda_(y[2], Lambda_);

                      //if (useScalarDissipation_) x[0] = min(pChi[facei], chiLimiter);
                      //if (useProgressVariable_) x[0] = min(Lambda, LambdaMax);		//added
                      if (useProgressVariable_) x[0] = Lambda;		//added
                      if (useMixtureFractionVariance_) x[1] = min(pZeta[facei], 0.99);
                      x[2] = pZ[facei];

                      ubP[facei] = solver.upperBounds(x);
                      posP[facei] = solver.position(ubP[facei], x);

                      pHe[facei] = solver.interpolate(ubP[facei], posP[facei], (solver.sizeTableNames() - 2));
                      pSrr[facei] = solver.interpolate(ubP[facei], posP[facei], (solver.sizeTableNames() - 1));
              	 }

             	 pY[facei] = solver.interpolate(ubP[facei], posP[facei], i);
              }
           }
        }

        // Calculate thermodynamic Properties
        thermo.correct();

        if (selectedFields.empty())
        {
        	forAll(Y, i)
            {
               Info << "Writing field " << thermo.composition().Y()[i].name() << endl;
        	   thermo.composition().Y()[i].write();
            }
            Info << "Writing field Zeta" << endl;
        	Zeta.write();
        }
        else
        {
        	forAll(Y, i)
            {
        	   if (selectedFields[thermo.composition().Y()[i].name()])
        	   {
                   Info << "Writing field " << thermo.composition().Y()[i].name() << endl;
        		   thermo.composition().Y()[i].write();
        	   }
            }
            Info << "Writing field Zeta" << endl;
        	Zeta.write();
        }

    }

	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
	Info<< "End\n" << endl;

	return 0;
}

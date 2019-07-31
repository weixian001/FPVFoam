
	
	turbulence->correctSrr();

	fvScalarMatrix PvEqn
	(
		(
		  fvm::ddt(rho, Chi)
		+ fvm::div(phi, Chi)
        	- fvm::laplacian(turbulence->DZEff(), Chi)
		==
		  thermo.Srr()
        	)
    	);
    	
    	
	PvEqn.relax();
//    	PvEqn.boundaryManipulate(Chi.boundaryField());
	PvEqn.solve(mesh.solver("Chi"));
	//bound(Chi,0);
	
	Info<< "----------> chi min/max   = " << min(Chi).value() << ", "
        << max(Chi).value() << endl;

//	Info << " div == " << fvm::div(phi, Chi) << endl;
//	Info << " laplacian == " << fvm::laplacian(turbulence->DZEff(), Chi) << endl;
//	Info << " fvm::ddt(rho, chi) = " << fvm::ddt(rho, Chi) << endl;
//	Info << "thermo.Srr()" << thermo.Srr() << endl;
//	Info << "solve = " <<( fvm::ddt(rho, Chi) + fvm::div(phi, Chi)- fvm::laplacian(turbulence->DZEff(), Chi)-thermo.Srr()) << endl;
//	Info << "solve " << PvEqn.solve(mesh.solver("Chi")) << endl;
//	Info << "solve " << PvEqn << endl;

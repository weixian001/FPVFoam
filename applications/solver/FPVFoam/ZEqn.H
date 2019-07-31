	fvScalarMatrix ZEqn
	(
		(
		  fvm::ddt(rho, Zmix)
		+ fvm::div(phi, Zmix)
        - fvm::laplacian(turbulence->DZEff(), Zmix)
        )
    );

	ZEqn.relax();
	ZEqn.solve(mesh.solver("Zmix"));
	//bound(Z,0);
    
    Info<< "----------> Zmix min/max   = " << min(Zmix).value() << ", "
    << max(Zmix).value() << endl;

	turbulence->correctVarZ();
	//turbulence->correctSrr();
	//turbulence->correctChi();
	
	
	

    combustion->correct();
    

# Numerical Simulation of a Fixed wing UAV and the effect of blended winglets  
The simulation setup consits of grid independence setup, turbulence sensitivity setup, UAV studies setup and validation setup.
Each of these folders contain sub-folders with all the required files to run the simulation. 
For each particular setup, there is a baseCase.  
The run script outside baseCase directory runs and generate the mesh for the UAV configuration that it is supposed to run. The stl file is same for the all the simulations except for cant45, and cant90 setup inside UAV studies folder.
The run script after generating mesh in baseCase copies the directory for running the solver.
The various angle of attack could be setup in two possible ways:
1. By rotating the geometry by the angle of attack while keeping the flow in the same direction.
2. By rotating the flow by the angle of attack while keeping the geometry unchanged.  
As rotating the geometry requires to generate new mesh for each angle of attack, this simulation instead rotates the flow and the direction of forces without transforming geometry and therefore mesh.  
As Angle of attack is the angle made by direction of flow with the chord of the wing of UAV and the wing is at 4 degree incidence to fuselage, at zero degree angle of attack flow needs to be at -4 degree with x axis.  
so velocity becomes (Vcos(alpha-4),Vsin(alpha-4),0). Here V = 20m/s
The lift direction and drag direction is adjusted in forceCoeffs file inside system directory. 
For this simulation setup, longitudinal axis is along x axis, lateral axis is along z axis and vertical axis is along y-axis.
As lift force is the force perpendicular to direction of flow and drag force is force along the flow, 
liftDir (-sin(alpha-4),cos(alpha-4),0);
dragDir (cos(alpha-4),sin(alpha-4),0);
After these 3 adjustments, we will get the flow in the required angle of attack without rotating geometry and thus without regenerating mesh.
Some python codes for generating plots are included in the proper directories.
The simulation has been tested in openFoam 2106 and 2406, and should work in other openFoam.com version as well.
Necessary geometry are included in geometry folder as well.
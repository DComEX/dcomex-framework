using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MGroup.Constitutive.Structural;
using MGroup.Constitutive.Structural.PlanarElements;
using MGroup.Environments;
using MGroup.FEM.Entities;
using MGroup.LinearAlgebra.Distributed.Overlapping;
using MGroup.MSolve.DataStructures;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.MSolve.Discretization.Loads;
using MGroup.Solvers.DDM.FetiDP.Dofs;
using MGroup.Solvers.DDM.Tests.Commons;
using MGroup.Solvers.Results;

// Global
// 72--73--74--75--76--77--78--79--80
//  |   |   |   |   |   |   |   |   |
// 63--64--65--66--67--68--69--70--71
//  |   |   |   |   |   |   |   |   |
// 54--55--56--57--58--59--60--61--62
//  |   |   |   |   |   |   |   |   |
// 45--46--47--48--49--50--51--52--53
//  |   |   |   |   |   |   |   |   |
// 36--37--38--39--40--41--42--43--44
//  |   |   |   |   |   |   |   |   |
// 27--28--29--30--31--32--33--34--35
//  |   |   |   |   |   |   |   |   |
// 18--19--20--21--22--23--24--25--26
//  |   |   |   |   |   |   |   |   |
// 09--10--11--12--13--14--15--16--17
//  |   |   |   |   |   |   |   |   |
// 00--01--02--03--04--05--06--07--08
//
// Boundary Conditions: Ux(0)=Uy(0)=Uy(8)=0, Px(80)=100
//
// ********************************************************
//
// Subdomains:
// 72--73--74    74--75--76    76--77--78    78--79--80
//  | 56| 57|     | 58| 59|     | 60| 61|     | 62| 63|
// 63--64--65    65--66--67    67--68--69    69--70--71
//  | 48| 49|     | 50| 51|     | 52| 53|     | 54| 55|
// 54--55--56    56--57--58    58--59--60    60--61--62
// s12           s13           s14           s15
//
// 54--55--56    56--57--58    58--59--60    60--61--62
//  | 40| 41|     | 42| 43|     | 44| 45|     | 46| 47|
// 45--46--47    47--48--49    49--50--51    51--52--53
//  | 32| 33|     | 34| 35|     | 36| 37|     | 38| 39|
// 36--37--38    38--39--40    40--41--42    42--43--44
// s8            s9            s10           s11
//
// 36--37--38    38--39--40    40--41--42    42--43--44
//  | 24| 25|     | 26| 27|     | 28| 29|     | 30| 31|
// 27--28--29    29--30--31    31--32--33    33--34--35
//  | 16| 17|     | 18| 19|     | 20| 21|     | 22| 23|
// 18--19--20    20--21--22    22--23--24    24--25--26
// s4            s5            s6            s7
//
// 18--19--20    20--21--22    22--23--24    24--25--26
//  | 8 | 9 |     | 10| 11|     | 12| 13|     | 14| 15|
// 09--10--11    11--12--13    13--14--15    15--16--17
//  | 0 | 1 |     | 2 | 3 |     | 4 | 5 |     | 6 | 7 |
// 00--01--02    02--03--04    04--05--06    06--07--08
// s0            s1            s2            s3
//
// ********************************************************
//
// Clusters:
// +---+---+    +---+---+
// |s12|s13|    |s14|s15|
// +---+---+    +---+---+
// | s8| s9|    |s10|s11|
// +---+---+    +---+---+
// c2           c3
//
// +---+---+    +---+---+
// | s4| s5|    | s6| s7|
// +---+---+    +---+---+
// | s0| s1|    | s2| s3|
// +---+---+    +---+---+
// c0           c1

namespace MGroup.Solvers.DDM.Tests.ExampleModels
{
	//TODOMPI: In this class the partitioning and subdomain topologies should be hardcoded. However also provide an automatic 
	//      way for 1D, 2D, 3D rectilinear meshes. Then use these hardcoded data for testing the automatic ones.
	//TODO: Add another row of clusters up top. Being symmetric is not good for tests, since a lot of mistakes are covered by the symmetry
	public class Plane2DExample
	{
		private const double E = 1.0, v = 0.3, thickness = 1.0;
		private const double load = 100;

		public static double[] MinCoords => new double[] { 0, 0 };

		public static double[] MaxCoords => new double[] { 8, 8 };

		public static int[] NumElements => new int[] { 8, 8 };

		public static int[] NumSubdomains => new int[] { 4, 4 };

		public static int[] NumClusters => new int[] { 2, 2 };

		public static void CheckDistributedIndexer(IComputeEnvironment environment, ComputeNodeTopology nodeTopology,
			DistributedOverlappingIndexer indexer)
		{
			//WARNING: Disable any other style analyzer here and DO NOT let them automatically format this
#pragma warning disable IDE0055
			Action<int> checkIndexer = subdomainID =>
			{
				int[] multiplicitiesExpected; // Remember that only boundary dofs go into the distributed vectors 
				var commonEntriesExpected = new Dictionary<int, int[]>();
				if (subdomainID == 0)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9
					// Boundary dofs:                      02x 02y 11x 11y 18x 18y 19x 19y 20x 20y
					multiplicitiesExpected =   new int[] {   2,  2,  2,  2,  2,  2,  2,  2,  4,  4 };
					commonEntriesExpected[1] = new int[] {   0,  1,  2,  3,                  8,  9 };
					commonEntriesExpected[4] = new int[] {                   4,  5,  6,  7,  8,  9 };
					commonEntriesExpected[5] = new int[] {                                   8,  9 };
				}
				else if (subdomainID == 1)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13
					// Boundary dofs:                      02x 02y 04x 04y 11x 11y 13x 13y 20x 20y 21x 21y 22x 22y
					multiplicitiesExpected =   new int[] {   2,  2,  2,  2,  2,  2,  2,  2,  4,  4,  2,  2,  4,  4 };
					commonEntriesExpected[0] = new int[] {   0,  1,          4,  5,          8,  9                 };
					commonEntriesExpected[2] = new int[] {           2,  3,          6,  7,                 12, 13 };
					commonEntriesExpected[4] = new int[] {                                   8,  9                 };
					commonEntriesExpected[5] = new int[] {                                  8,  9,  10, 11, 12, 13 };
					commonEntriesExpected[6] = new int[] {                                                  12, 13 };
				}
				else if (subdomainID == 2)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13
					// Boundary dofs:                      04x 04y 06x 06y 13x 13y 15x 15y 22x 22y 23x 23y 24x 24y
					multiplicitiesExpected =   new int[] {   2,  2,  2,  2,  2,  2,  2,  2,  4,  4,  2,  2,  4,  4 };
					commonEntriesExpected[1] = new int[] {   0,  1,          4,  5,          8,  9                 };
					commonEntriesExpected[3] = new int[] {           2,  3,          6,  7,                 12, 13 };
					commonEntriesExpected[5] = new int[] {                                   8,  9                 };
					commonEntriesExpected[6] = new int[] {                                   8,  9, 10, 11, 12, 13 };
					commonEntriesExpected[7] = new int[] {                                                  12, 13 };
				}
				else if (subdomainID == 3)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9
					// Boundary dofs:                      06x 06y 15x 15y 24x 24y 25x 25y 26x 26y
					multiplicitiesExpected =   new int[] {   2,  2,  2,  2,  4,  4,  2,  2,  2,  2 };
					commonEntriesExpected[2] = new int[] {   0,  1,  2,  3,  4,  5                 };
					commonEntriesExpected[6] = new int[] {                   4,  5,                };
					commonEntriesExpected[7] = new int[] {                   4,  5,  6,  7,  8,  9 };
				}
				else if (subdomainID == 4)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13
					// Boundary dofs:                      18x 18y 19x 19y 20x 20y 29x 29y 36x 36y 37x 37y 38x 38y
					multiplicitiesExpected =   new int[] {   2,  2,  2,  2,  4,  4,  2,  2,  2,  2,  2,  2,  4,  4 };
					commonEntriesExpected[0] = new int[] {   0,  1,  2,  3,  4,  5                                 };
					commonEntriesExpected[1] = new int[] {                   4,  5                                 };
					commonEntriesExpected[5] = new int[] {                   4,  5,  6,  7,                 12, 13 };
					commonEntriesExpected[8] = new int[] {                                   8,  9, 10, 11, 12, 13 };
					commonEntriesExpected[9] = new int[] {                                                  12, 13 };
				}
				else if (subdomainID == 5)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  
					// Boundary dofs:                      20x 20y 21x 21y 22x 22y 29x 29y 31x 31y 38x 38y 39x 39y 40x 40y
					multiplicitiesExpected =    new int[] {  4,  4,  2,  2,  4,  4,  2,  2,  2,  2,  4,  4,  2,  2,  4,  4 };
					commonEntriesExpected[0] =  new int[] {  0,  1                                                         };
					commonEntriesExpected[1] =  new int[] {  0,  1,  2,  3,  4,  5                                         };
					commonEntriesExpected[2] =  new int[] {                  4,  5                                         };
					commonEntriesExpected[4] =  new int[] {  0,  1,                  6,  7,         10, 11                 };
					commonEntriesExpected[6] =  new int[] {                  4,  5,          8,  9,                 14, 15 };
					commonEntriesExpected[8] =  new int[] {                                         10, 11                 };
					commonEntriesExpected[9] =  new int[] {                                         10, 11, 12, 13, 14, 15 };
					commonEntriesExpected[10] = new int[] {                                                         14, 15 };
				}
				else if (subdomainID == 6)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  
					// Boundary dofs:                      22x 22y 23x 23y 24x 24y 31x 31y 33x 33y 40x 40y 41x 41y 42x 42y
					multiplicitiesExpected =    new int[] {  4,  4,  2,  2,  4,  4,  2,  2,  2,  2,  4,  4,  2,  2,  4,  4 };
					commonEntriesExpected[1] =  new int[] {  0,  1                                                         };
					commonEntriesExpected[2] =  new int[] {  0,  1,  2,  3,  4,  5                                         };
					commonEntriesExpected[3] =  new int[] {                  4,  5                                         };
					commonEntriesExpected[5] =  new int[] {  0,  1,                  6,  7,         10, 11                 };
					commonEntriesExpected[7] =  new int[] {                  4,  5,          8,  9,                 14, 15 };
					commonEntriesExpected[9] =  new int[] {                                         10, 11                 };
					commonEntriesExpected[10] = new int[] {                                         10, 11, 12, 13, 14, 15 };
					commonEntriesExpected[11] = new int[] {                                                         14, 15 };
				}
				else if (subdomainID == 7)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13
					// Boundary dofs:                      24x 24y 25x 25y 26x 26y 33x 33y 42x 42y 43x 43y 44x 44y
					multiplicitiesExpected =    new int[] {  4,  4,  2,  2,  2,  2,  2,  2,  4,  4,  2,  2,  2,  2 };
					commonEntriesExpected[2] =  new int[] {  0,  1                                                 };
					commonEntriesExpected[3] =  new int[] {  0,  1,  2,  3,  4,  5                                 };
					commonEntriesExpected[6] =  new int[] {  0,  1,                  6,  7,  8,  9                 };
					commonEntriesExpected[10] = new int[] {                                  8,  9                 };
					commonEntriesExpected[11] = new int[] {                                  8,  9, 10, 11, 12, 13 };
				}
				else if (subdomainID == 8)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13
					// Boundary dofs:                      36x 36y 37x 37y 38x 38y 47x 47y 54x 54y 55x 55y 56x 56y
					multiplicitiesExpected =    new int[] {  2,  2,  2,  2,  4,  4,  2,  2,  2,  2,  2,  2,  4,  4 };
					commonEntriesExpected[4] =  new int[] {  0,  1,  2,  3,  4,  5                                 };
					commonEntriesExpected[5] =  new int[] {                  4,  5                                 };
					commonEntriesExpected[9] =  new int[] {                  4,  5,  6,  7,                  12, 13 };
					commonEntriesExpected[12] = new int[] {                                   8,  9, 10, 11, 12, 13 };
					commonEntriesExpected[13] = new int[] {                                                  12, 13 };
				}
				else if (subdomainID == 9)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  
					// Boundary dofs:                      38x 38y 39x 39y 40x 40y 47x 47y 49x 49y 56x 56y 57x 57y 58x 58y
					multiplicitiesExpected =    new int[] {  4,  4,  2,  2,  4,  4,  2,  2,  2,  2,  4,  4,  2,  2,  4,  4 };
					commonEntriesExpected[4] =  new int[] {  0,  1                                                         };
					commonEntriesExpected[5] =  new int[] {  0,  1,  2,  3,  4,  5                                         };
					commonEntriesExpected[6] =  new int[] {                  4,  5                                         };
					commonEntriesExpected[8] =  new int[] {  0,  1,                  6,  7,         10, 11                 };
					commonEntriesExpected[10] = new int[] {                  4,  5,          8,  9,                 14, 15 };
					commonEntriesExpected[12] = new int[] {                                         10, 11                 };
					commonEntriesExpected[13] = new int[] {                                         10, 11, 12, 13, 14, 15 };
					commonEntriesExpected[14] = new int[] {                                                         14, 15 };
				}
				else if (subdomainID == 10)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  
					// Boundary dofs:                      40x 40y 41x 41y 42x 42y 49x 49y 51x 51y 58x 58y 59x 59y 60x 60y
					multiplicitiesExpected =    new int[] {  4,  4,  2,  2,  4,  4,  2,  2,  2,  2,  4,  4,  2,  2,  4,  4 };
					commonEntriesExpected[5] =  new int[] {  0,  1                                                         };
					commonEntriesExpected[6] =  new int[] {  0,  1,  2,  3,  4,  5                                         };
					commonEntriesExpected[7] =  new int[] {                  4,  5                                         };
					commonEntriesExpected[9] =  new int[] {  0,  1,                  6,  7,         10, 11                 };
					commonEntriesExpected[11] = new int[] {                  4,  5,          8,  9,                 14, 15 };
					commonEntriesExpected[13] = new int[] {                                         10, 11                 };
					commonEntriesExpected[14] = new int[] {                                         10, 11, 12, 13, 14, 15 };
					commonEntriesExpected[15] = new int[] {                                                         14, 15 };
				}
				else if (subdomainID == 11)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13
					// Boundary dofs:                      42x 42y 43x 43y 44x 44y 51x 51y 60x 60y 61x 61y 62x 62y
					multiplicitiesExpected =    new int[] {  4,  4,  2,  2,  2,  2,  2,  2,  4,  4,  2,  2,  2,  2 };
					commonEntriesExpected[6] =  new int[] {  0,  1                                                 };
					commonEntriesExpected[7] =  new int[] {  0,  1,  2,  3,  4,  5                                 };
					commonEntriesExpected[10] = new int[] {  0,  1,                  6,  7,  8,  9                 };
					commonEntriesExpected[14] = new int[] {                                  8,  9                 };
					commonEntriesExpected[15] = new int[] {                                  8,  9, 10, 11, 12, 13 };
				}
				else if (subdomainID == 12)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9
					// Boundary dofs:                      54x 54y 55x 55y 56x 56y 65x 65y 74x 74y
					multiplicitiesExpected =    new int[] {  2,  2,  2,  2,  4,  4,  2,  2,  2,  2 };
					commonEntriesExpected[8] =  new int[] {  0,  1,  2,  3,  4,  5 };
					commonEntriesExpected[9] =  new int[] {                  4,  5 };
					commonEntriesExpected[13] = new int[] {                  4,  5,  6,  7,  8,  9 };
				}
				else if (subdomainID == 13)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13
					// Boundary dofs:                      56x 56y 57x 57y 58x 58y 65x 65y 67x 67y 74x 74y 76x 76y
					multiplicitiesExpected =    new int[] {  4,  4,  2,  2,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2 };
					commonEntriesExpected[8] =  new int[] {  0,  1                                                 };
					commonEntriesExpected[9] =  new int[] {  0,  1,  2,  3,  4,  5                                 };
					commonEntriesExpected[10] = new int[] {                  4,  5                                 };
					commonEntriesExpected[12] = new int[] {  0,  1,                  6,  7,         10, 11         };
					commonEntriesExpected[14] = new int[] {                  4,  5,          8,  9,         12, 13 };
				}
				else if (subdomainID == 14)
				{
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9  10  11  12  13
					// Boundary dofs:                      58x 58y 59x 59y 60x 60y 67x 67y 69x 69y 76x 76y 78x 78y
					multiplicitiesExpected =    new int[] {  4,  4,  2,  2,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2 };
					commonEntriesExpected[9] =  new int[] {  0,  1                                                 };
					commonEntriesExpected[10] = new int[] {  0,  1,  2,  3,  4,  5                                 };
					commonEntriesExpected[11] = new int[] {                  4,  5                                 };
					commonEntriesExpected[13] = new int[] {  0,  1,                  6,  7,         10, 11         };
					commonEntriesExpected[15] = new int[] {                  4,  5,          8,  9,         12, 13 };
				}
				else
				{
					Debug.Assert(subdomainID == 15);
					// Boundary dof idx:                     0   1   2   3   4   5   6   7   8   9
					// Boundary dofs:                      60x 60y 6x1 61y 62x 62y 69x 69y 78x 78y
					multiplicitiesExpected =    new int[] {  4,  4,  2,  2,  2,  2,  2,  2,  2,  2 };
					commonEntriesExpected[10] = new int[] {  0,  1                                 };
					commonEntriesExpected[11] = new int[] {  0,  1,  2,  3,  4,  5                 };
					commonEntriesExpected[14] = new int[] {  0,  1,                  6,  7,  8,  9 };
				}

				double[] inverseMultiplicities = indexer.GetLocalComponent(subdomainID).InverseMultiplicities;
				var multiplicitiesComputed = new int[inverseMultiplicities.Length];
				for (int i = 0; i < inverseMultiplicities.Length; ++i)
				{
					multiplicitiesComputed[i] = (int)Math.Round(1.0 / inverseMultiplicities[i]);
				}
				if (!Utilities.AreEqual(multiplicitiesExpected, multiplicitiesComputed))
					Console.WriteLine("Error in CheckDistributedIndexer");
				foreach (int neighborID in commonEntriesExpected.Keys)
				{
					int[] expected = commonEntriesExpected[neighborID];
					int[] computed = indexer.GetLocalComponent(subdomainID).GetCommonEntriesWithNeighbor(neighborID);
					if(!Utilities.AreEqual(expected, computed))
						Console.WriteLine("Error in CheckDistributedIndexer");
				}
			};
			environment.DoPerNode(checkIndexer);
#pragma warning restore IDE0055
		}

		public static ComputeNodeTopology CreateNodeTopology()
		{
			var nodeTopology = new ComputeNodeTopology();
			Dictionary<int, int> clustersOfSubdomains = GetSubdomainClusters();
			Dictionary<int, int[]> neighborsOfSubdomains = GetSubdomainNeighbors();
			for (int s = 0; s < NumSubdomains[0] * NumSubdomains[1]; ++s)
			{
				nodeTopology.AddNode(s, neighborsOfSubdomains[s], clustersOfSubdomains[s]);
			}
			return nodeTopology;
		}

		public static Model CreateSingleSubdomainModel()
		{
			var builder = new UniformDdmModelBuilder2D();
			builder.MinCoords = MinCoords;
			builder.MaxCoords = MaxCoords;
			builder.NumElementsTotal = NumElements;
			builder.NumSubdomains = NumSubdomains;
			builder.NumClusters = NumClusters;
			builder.MaterialHomogeneous = new ElasticMaterial2D(StressState2D.PlaneStress) { YoungModulus = E, PoissonRatio = v };
			Model model = builder.BuildSingleSubdomainModel();

			// Boundary conditions
			model.NodesDictionary[0].Constraints.Add(new Constraint() { DOF = StructuralDof.TranslationX, Amount = 0 });
			model.NodesDictionary[0].Constraints.Add(new Constraint() { DOF = StructuralDof.TranslationY, Amount = 0 });
			model.NodesDictionary[8].Constraints.Add(new Constraint() { DOF = StructuralDof.TranslationY, Amount = 0 });
			model.Loads.Add(new Load() { Node = model.NodesDictionary[80], DOF = StructuralDof.TranslationX, Amount = load });

			return model;
		}

		public static IModel CreateMultiSubdomainModel()
		{
			Dictionary<int, int> elementsToSubdomains = GetSubdomainsOfElements();
			Model model = CreateSingleSubdomainModel();
			model.DecomposeIntoSubdomains(NumSubdomains[0] * NumSubdomains[1], e => elementsToSubdomains[e]);
			return model;
		}

		public static ICornerDofSelection GetCornerDofs(IModel model)
		{
			//int[] cornerNodes = { 20, 22, 24, 38, 40, 42, 56, 58, 60 };
			return UniformDdmModelBuilder2D.FindCornerDofs(model, 2);
		}

		public static NodalResults GetExpectedNodalValues(ActiveDofs allDofs)
		{
			int dofX = allDofs.GetIdOfDof(StructuralDof.TranslationX);
			int dofY = allDofs.GetIdOfDof(StructuralDof.TranslationY);

			var results = new Table<int, int, double>();
			#region long list of solution values per dof
			results[0, dofX] = 0;
			results[0, dofY] = 0;
			results[1, dofX] = 150.1000802198485;
			results[1, dofY] = 66.0931314769534;
			results[2, dofX] = 209.1956299086393;
			results[2, dofY] = 4.108111394651694;
			results[3, dofX] = 244.30619107657705;
			results[3, dofY] = -65.70787578919311;
			results[4, dofX] = 275.5145123577474;
			results[4, dofY] = -143.7789503887835;
			results[5, dofX] = 310.9820696441611;
			results[5, dofY] = -219.0962997007431;
			results[6, dofX] = 360.58357755214274;
			results[6, dofY] = -275.2450418749667;
			results[7, dofX] = 434.3287503604323;
			results[7, dofY] = -289.7137836350665;
			results[8, dofX] = 515.7013066725633;
			results[8, dofY] = 0;
			results[9, dofX] = 272.6359049640539;
			results[9, dofY] = 150.08004029482072;
			results[10, dofX] = 252.78020747268184;
			results[10, dofY] = 46.32961764070971;
			results[11, dofX] = 293.8503312870056;
			results[11, dofY] = -1.7443278815822723;
			results[12, dofX] = 326.4095289392989;
			results[12, dofY] = -73.49070643258257;
			results[13, dofX] = 351.75879754287325;
			results[13, dofY] = -150.6195892651938;
			results[14, dofX] = 371.86509727481035;
			results[14, dofY] = -225.85849979395613;
			results[15, dofX] = 385.24033375505536;
			results[15, dofY] = -286.23854752027336;
			results[16, dofX] = 378.3432980307993;
			results[16, dofY] = -286.3224550149545;
			results[17, dofX] = 394.9484813476963;
			results[17, dofY] = -232.27660571031268;
			results[18, dofX] = 417.54856262267583;
			results[18, dofY] = 208.80895110877685;
			results[19, dofX] = 411.2439064986351;
			results[19, dofY] = 87.80601372931197;
			results[20, dofX] = 412.3702182134087;
			results[20, dofY] = 2.2745995313056504;
			results[21, dofX] = 426.3131179538608;
			results[21, dofY] = -72.66524240626192;
			results[22, dofX] = 437.3854495891751;
			results[22, dofY] = -150.17089381009788;
			results[23, dofX] = 441.5315009766441;
			results[23, dofY] = -224.82357750196996;
			results[24, dofX] = 436.4711271034646;
			results[24, dofY] = -284.59569325130064;
			results[25, dofX] = 433.7125389107416;
			results[25, dofY] = -320.6803824770333;
			results[26, dofX] = 451.79182247473295;
			results[26, dofY] = -368.15093155855277;
			results[27, dofX] = 555.699870036769;
			results[27, dofY] = 241.5055394213765;
			results[28, dofX] = 547.6227125634863;
			results[28, dofY] = 120.65889928467423;
			results[29, dofX] = 543.3539929002895;
			results[29, dofY] = 22.465613854123653;
			results[30, dofX] = 542.5266903063331;
			results[30, dofY] = -62.36662223115644;
			results[31, dofX] = 542.0039823977013;
			results[31, dofY] = -142.261841244437;
			results[32, dofX] = 537.5778988014689;
			results[32, dofY] = -219.53304085182225;
			results[33, dofX] = 531.8986496143507;
			results[33, dofY] = -292.48395371557837;
			results[34, dofX] = 533.8423366603457;
			results[34, dofY] = -365.4661452270108;
			results[35, dofX] = 549.5083401123986;
			results[35, dofY] = -456.99921095913686;
			results[36, dofX] = 685.5850587063886;
			results[36, dofY] = 264.12566942589245;
			results[37, dofX] = 679.9174122725947;
			results[37, dofY] = 144.87822569894286;
			results[38, dofX] = 674.4962201087524;
			results[38, dofY] = 42.86748064528494;
			results[39, dofX] = 670.1190505007278;
			results[39, dofY] = -46.773800599539214;
			results[40, dofX] = 665.2183523624371;
			results[40, dofY] = -130.41758856464403;
			results[41, dofX] = 658.7853356236066;
			results[41, dofY] = -214.0613765297487;
			results[42, dofX] = 653.5483942047391;
			results[42, dofY] = -303.7026577745726;
			results[43, dofX] = 655.7711373677146;
			results[43, dofY] = -405.7134028282301;
			results[44, dofX] = 669.1551394111781;
			results[44, dofY] = -524.9608465551795;
			results[45, dofX] = 812.6701113852116;
			results[45, dofY] = 278.4034109056682;
			results[46, dofX] = 809.3317700635641;
			results[46, dofY] = 161.50160672706784;
			results[47, dofX] = 806.2763156728598;
			results[47, dofY] = 59.068234606991794;
			results[48, dofX] = 804.1683684889136;
			results[48, dofY] = -32.50837197199394;
			results[49, dofX] = 802.3182675762101;
			results[49, dofY] = -118.57333588485118;
			results[50, dofX] = 799.2195769840496;
			results[50, dofY] = -207.26231920360337;
			results[51, dofX] = 794.8209723869212;
			results[51, dofY] = -310.72024900411236;
			results[52, dofX] = 795.5513941604238;
			results[52, dofY] = -438.36471504330586;
			results[53, dofX] = 806.4785814608416;
			results[53, dofY] = -584.5800936264815;
			results[54, dofX] = 935.5890229324615;
			results[54, dofY] = 285.093677831206;
			results[55, dofX] = 934.7123291228971;
			results[55, dofY] = 171.3381585281947;
			results[56, dofX] = 935.3863653899402;
			results[56, dofY] = 69.07779838031117;
			results[57, dofX] = 939.1478145808935;
			results[57, dofY] = -23.49039638687864;
			results[58, dofX] = 946.0190333867533;
			results[58, dofY] = -110.6642833191903;
			results[59, dofX] = 954.3661976036765;
			results[59, dofY] = -200.69113796346565;
			results[60, dofX] = 959.4872742799955;
			results[60, dofY] = -308.42705891889176;
			results[61, dofX] = 957.1809615350032;
			results[61, dofY] = -460.13414403904795;
			results[62, dofX] = 969.8322827845182;
			results[62, dofY] = -647.4220516400029;
			results[63, dofX] = 1052.2061027387206;
			results[63, dofY] = 286.3804712031824;
			results[64, dofX] = 1052.9250328911655;
			results[64, dofY] = 175.49596313767336;
			results[65, dofX] = 1056.6765652263657;
			results[65, dofY] = 72.64445034546856;
			results[66, dofX] = 1066.2194291244234;
			results[66, dofY] = -21.451667088065413;
			results[67, dofX] = 1083.9727523418535;
			results[67, dofY] = -110.21558786409435;
			results[68, dofX] = 1111.6749974599336;
			results[68, dofY] = -200.86948094397187;
			results[69, dofX] = 1148.0665676944136;
			results[69, dofY] = -306.3319292021886;
			results[70, dofX] = 1178.4881234492807;
			results[70, dofY] = -457.1734800220033;
			results[71, dofX] = 1174.5186791223607;
			results[71, dofY] = -725.8542600462622;
			results[72, dofX] = 1163.4194269822522;
			results[72, dofY] = 286.15284032333284;
			results[73, dofX] = 1163.2118360274305;
			results[73, dofY] = 176.30707694524676;
			results[74, dofX] = 1164.8652682742418;
			results[74, dofY] = 72.21997338371091;
			results[75, dofX] = 1173.9695080551178;
			results[75, dofY] = -24.49568157981949;
			results[76, dofX] = 1196.835440811547;
			results[76, dofY] = -117.05622674050431;
			results[77, dofX] = 1240.6453866227;
			results[77, dofY] = -212.37049718881994;
			results[78, dofX] = 1316.2532159177426;
			results[78, dofY] = -322.75339716197135;
			results[79, dofX] = 1447.4405061680093;
			results[79, dofY] = -474.35677904570855;
			results[80, dofX] = 1679.120733654807;
			results[80, dofY] = -807.8231945819023;
			#endregion

			return new NodalResults(results);
		}

		public static Dictionary<int, int> GetSubdomainsOfElements()
		{
			var elementsToSubdomains = new Dictionary<int, int>();
			#region long list of element -> subdomain associations
			elementsToSubdomains[0] = 0;
			elementsToSubdomains[1] = 0;
			elementsToSubdomains[2] = 1;
			elementsToSubdomains[3] = 1;
			elementsToSubdomains[4] = 2;
			elementsToSubdomains[5] = 2;
			elementsToSubdomains[6] = 3;
			elementsToSubdomains[7] = 3;
			elementsToSubdomains[8] = 0;
			elementsToSubdomains[9] = 0;
			elementsToSubdomains[10] = 1;
			elementsToSubdomains[11] = 1;
			elementsToSubdomains[12] = 2;
			elementsToSubdomains[13] = 2;
			elementsToSubdomains[14] = 3;
			elementsToSubdomains[15] = 3;
			elementsToSubdomains[16] = 4;
			elementsToSubdomains[17] = 4;
			elementsToSubdomains[18] = 5;
			elementsToSubdomains[19] = 5;
			elementsToSubdomains[20] = 6;
			elementsToSubdomains[21] = 6;
			elementsToSubdomains[22] = 7;
			elementsToSubdomains[23] = 7;
			elementsToSubdomains[24] = 4;
			elementsToSubdomains[25] = 4;
			elementsToSubdomains[26] = 5;
			elementsToSubdomains[27] = 5;
			elementsToSubdomains[28] = 6;
			elementsToSubdomains[29] = 6;
			elementsToSubdomains[30] = 7;
			elementsToSubdomains[31] = 7;
			elementsToSubdomains[32] = 8;
			elementsToSubdomains[33] = 8;
			elementsToSubdomains[34] = 9;
			elementsToSubdomains[35] = 9;
			elementsToSubdomains[36] = 10;
			elementsToSubdomains[37] = 10;
			elementsToSubdomains[38] = 11;
			elementsToSubdomains[39] = 11;
			elementsToSubdomains[40] = 8;
			elementsToSubdomains[41] = 8;
			elementsToSubdomains[42] = 9;
			elementsToSubdomains[43] = 9;
			elementsToSubdomains[44] = 10;
			elementsToSubdomains[45] = 10;
			elementsToSubdomains[46] = 11;
			elementsToSubdomains[47] = 11;
			elementsToSubdomains[48] = 12;
			elementsToSubdomains[49] = 12;
			elementsToSubdomains[50] = 13;
			elementsToSubdomains[51] = 13;
			elementsToSubdomains[52] = 14;
			elementsToSubdomains[53] = 14;
			elementsToSubdomains[54] = 15;
			elementsToSubdomains[55] = 15;
			elementsToSubdomains[56] = 12;
			elementsToSubdomains[57] = 12;
			elementsToSubdomains[58] = 13;
			elementsToSubdomains[59] = 13;
			elementsToSubdomains[60] = 14;
			elementsToSubdomains[61] = 14;
			elementsToSubdomains[62] = 15;
			elementsToSubdomains[63] = 15;
			#endregion

			return elementsToSubdomains;
		}

		public static Dictionary<int, int> GetSubdomainClusters()
		{
			var result = new Dictionary<int, int>();

			result[ 0] = 0;
			result[ 1] = 0;
			result[ 2] = 1;
			result[ 3] = 1;
			result[ 4] = 0;
			result[ 5] = 0;
			result[ 6] = 1;
			result[ 7] = 1;
			result[ 8] = 2;
			result[ 9] = 2;
			result[10] = 3;
			result[11] = 3;
			result[12] = 2;
			result[13] = 2;
			result[14] = 3;
			result[15] = 3;

			return result;
		}

		public static Dictionary<int, int[]> GetSubdomainNeighbors()
		{
			var result = new Dictionary<int, int[]>();

			result[ 0] =  new int[] { 1, 4, 5 };
			result[ 1] =  new int[] { 0, 2, 4, 5, 6 };
			result[ 2] =  new int[] { 1, 3, 5, 6, 7 };
			result[ 3] =  new int[] { 2, 6, 7 };
			result[ 4] =  new int[] { 0, 1, 5, 8, 9 };
			result[ 5] =  new int[] { 0, 1, 2, 4, 6, 8, 9, 10 };
			result[ 6] =  new int[] { 1, 2, 3, 5, 7, 9, 10, 11 };
			result[ 7] =  new int[] { 2, 3, 6, 10, 11 };
			result[ 8] =  new int[] { 4, 5, 9, 12, 13 };
			result[ 9] =  new int[] { 4, 5, 6, 8, 10, 12, 13, 14 };
			result[10] =  new int[] { 5, 6, 7, 9, 11, 13, 14, 15 };
			result[11] =  new int[] { 6, 7, 10, 14, 15 };
			result[12] =  new int[] { 8, 9, 13 };
			result[13] =  new int[] { 8, 9, 10, 12, 14 };
			result[14] =  new int[] { 9, 10, 11, 13, 15 };
			result[15] =  new int[] { 10, 11, 14 };

			return result;
		}
	}
}

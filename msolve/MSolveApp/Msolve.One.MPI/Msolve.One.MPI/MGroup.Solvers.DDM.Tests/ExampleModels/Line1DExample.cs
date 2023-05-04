using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MGroup.Constitutive.Thermal;
using MGroup.Environments;
using MGroup.FEM.Entities;
using MGroup.FEM.Thermal.Elements;
using MGroup.LinearAlgebra.Distributed.Overlapping;
using MGroup.MSolve.DataStructures;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.MSolve.Discretization.Loads;
using MGroup.Solvers;
using MGroup.Solvers.DDM.Tests.Commons;
using MGroup.Solvers.Results;

// Global:
// 0--1--2--3--4--5--6--7--8--9--10--11--12--13--14--15--[16] -> 16 is constrained, 0 is loaded
//
// Clusters:
// 0--1--2--3--4  c0
//             4--5--6--7--8  c1
//                         8--9--10--11--12  c2
//                                       12--13--14--15--[16]  c3
//
// Subdomains:
// 0--1--2  s0
//       2--3--4  s1
//             4--5--6  s2
//                   6--7--8  s3
//                         8--9--10  s4
//                               10--11--12  s5
//                                       12--13--14  s6
//                                               14--15--[16]  s7
//
namespace MGroup.Solvers.DDM.Tests.ExampleModels
{
	public class Line1DExample
	{
		private const double length = 2.0, sectionArea = 1.0;
		private const double conductivity = 1.0, specialHeat = 1.0, density = 1.0;

		public const int NumSubdomains = 8;

		//TODOMPI: It would be better if I could have a mock indexer object which knows how to compare itself with the actual one.
		public static void CheckDistributedIndexer(IComputeEnvironment environment, ComputeNodeTopology nodeTopology,
			DistributedOverlappingIndexer indexer)
		{
			Action<int> checkIndexer = subdomainID =>
			{
				int[] multiplicitiesExpected; // Remember that only boundary dofs go into the distributed vectors 
				var commonEntriesExpected = new Dictionary<int, int[]>();
				if (subdomainID == 0)
				{
					multiplicitiesExpected = new int[] { 2 };
					commonEntriesExpected[1] = new int[] { 0 };
				}
				else if (subdomainID == 1)
				{
					multiplicitiesExpected = new int[] { 2, 2 };
					commonEntriesExpected[0] = new int[] { 0 };
					commonEntriesExpected[2] = new int[] { 1 };
				}
				else if (subdomainID == 2)
				{
					multiplicitiesExpected = new int[] { 2, 2 };
					commonEntriesExpected[1] = new int[] { 0 };
					commonEntriesExpected[3] = new int[] { 1 };
				}
				else if (subdomainID == 3)
				{
					multiplicitiesExpected = new int[] { 2, 2 };
					commonEntriesExpected[2] = new int[] { 0 };
					commonEntriesExpected[4] = new int[] { 1 };
				}
				else if (subdomainID == 4)
				{
					multiplicitiesExpected = new int[] { 2, 2 };
					commonEntriesExpected[3] = new int[] { 0 };
					commonEntriesExpected[5] = new int[] { 1 };
				}
				else if (subdomainID == 5)
				{
					multiplicitiesExpected = new int[] { 2, 2 };
					commonEntriesExpected[4] = new int[] { 0 };
					commonEntriesExpected[6] = new int[] { 1 };
				}
				else if (subdomainID == 6)
				{
					multiplicitiesExpected = new int[] { 2, 2 };
					commonEntriesExpected[5] = new int[] { 0 };
					commonEntriesExpected[7] = new int[] { 1 };
				}
				else
				{
					Debug.Assert(subdomainID == 7);
					multiplicitiesExpected = new int[] { 2 };
					commonEntriesExpected[6] = new int[] { 0 };
				}

				double[] inverseMultiplicities = indexer.GetLocalComponent(subdomainID).InverseMultiplicities;
				var multiplicitiesComputed = new int[inverseMultiplicities.Length];
				for (int i = 0; i < inverseMultiplicities.Length; ++i)
				{
					multiplicitiesComputed[i] = (int)Math.Round(1.0 / inverseMultiplicities[i]);
				}

				if(!Utilities.AreEqual(multiplicitiesExpected, multiplicitiesComputed))
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
		}

		public static ComputeNodeTopology CreateNodeTopology()
		{
			var nodeTopology = new ComputeNodeTopology();
			nodeTopology.AddNode(0, new int[] { 1 }, 0);
			nodeTopology.AddNode(1, new int[] { 0, 2 }, 0);
			nodeTopology.AddNode(2, new int[] { 1, 3 }, 1);
			nodeTopology.AddNode(3, new int[] { 2, 4 }, 1);
			nodeTopology.AddNode(4, new int[] { 3, 5 }, 2);
			nodeTopology.AddNode(5, new int[] { 4, 6 }, 2);
			nodeTopology.AddNode(6, new int[] { 5, 7 }, 3);
			nodeTopology.AddNode(7, new int[] { 6 }, 3);

			return nodeTopology;
		}

		public static Model CreateSingleSubdomainModel()
		{
			var model = new Model();
			model.SubdomainsDictionary[0] = new Subdomain(0);

			// Nodes
			for (int n = 0; n <= 16; ++n)
			{
				model.NodesDictionary[n] = new Node(n, n * length, 0.0, 0.0);
			}

			// Materials
			var material = new ThermalMaterial(density, specialHeat, conductivity);

			// Elements
			for (int e = 0; e < 16; ++e)
			{
				Node[] nodes = { model.NodesDictionary[e], model.NodesDictionary[e + 1] };
				var elementType = new ThermalRod(nodes, sectionArea, material);
				var element = new Element() { ID = e, ElementType = elementType };
				foreach (var node in nodes) element.AddNode(node);
				model.ElementsDictionary[element.ID] = element;
				model.SubdomainsDictionary[0].Elements.Add(element);
			}

			// Boundary conditions
			model.NodesDictionary[16].Constraints.Add(new Constraint() { DOF = ThermalDof.Temperature, Amount = 0 });
			model.Loads.Add(new Load() { Node = model.NodesDictionary[0], DOF = ThermalDof.Temperature, Amount = 1 });

			return model;
		}

		public static IModel CreateMultiSubdomainModel()
		{
			// Partition
			Model model = CreateSingleSubdomainModel();
			var elementsToSubdomains = new Dictionary<int, int>();
			elementsToSubdomains[0] = 0;
			elementsToSubdomains[1] = 0;
			elementsToSubdomains[2] = 1;
			elementsToSubdomains[3] = 1;
			elementsToSubdomains[4] = 2;
			elementsToSubdomains[5] = 2;
			elementsToSubdomains[6] = 3;
			elementsToSubdomains[7] = 3;
			elementsToSubdomains[8] = 4;
			elementsToSubdomains[9] = 4;
			elementsToSubdomains[10] = 5;
			elementsToSubdomains[11] = 5;
			elementsToSubdomains[12] = 6;
			elementsToSubdomains[13] = 6;
			elementsToSubdomains[14] = 7;
			elementsToSubdomains[15] = 7;
			model.DecomposeIntoSubdomains(8, e => elementsToSubdomains[e]);

			return model;
		}

		public static NodalResults GetExpectedNodalValues(ActiveDofs allDofs)
		{
			//var model = Line1DExample.CreateSingleSubdomainModel();
			//var solver = new ISAAR.MSolve.Solvers.Direct.SkylineSolver.Builder().BuildSolver(model);
			//var problem = new ISAAR.MSolve.Problems.ProblemThermalSteadyState(model, solver);
			//var childAnalyzer = new ISAAR.MSolve.Analyzers.LinearAnalyzer(model, solver, problem);
			//var parentAnalyzer = new ISAAR.MSolve.Analyzers.StaticAnalyzer(model, solver, problem, childAnalyzer);
			//parentAnalyzer.Initialize();
			//parentAnalyzer.Solve();
			//Table<int, int, double> result =
			//    Utilities.FindNodalFieldValues(model.Subdomains.First(), solver.LinearSystems.First().Value.Solution);

			int dofID = allDofs.GetIdOfDof(ThermalDof.Temperature);
			var results = new Table<int, int, double>();
			results[0, dofID] = 32;
			results[1, dofID] = 30;
			results[2, dofID] = 28;
			results[3, dofID] = 26;
			results[4, dofID] = 24;
			results[5, dofID] = 22;
			results[6, dofID] = 20;
			results[7, dofID] = 18;
			results[8, dofID] = 16;
			results[9, dofID] = 14;
			results[10, dofID] = 12;
			results[11, dofID] = 10;
			results[12, dofID] = 8;
			results[13, dofID] = 6;
			results[14, dofID] = 4;
			results[15, dofID] = 2;
			results[16, dofID] = 0;

			return new NodalResults(results);
		}
	}
}

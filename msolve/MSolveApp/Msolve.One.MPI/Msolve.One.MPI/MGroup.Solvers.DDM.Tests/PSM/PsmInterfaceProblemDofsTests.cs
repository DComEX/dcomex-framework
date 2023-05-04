using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MGroup.Environments;
using MGroup.Environments.Mpi;
using MGroup.LinearAlgebra.Distributed.Overlapping;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MSolve.Discretization;
using MGroup.Solvers.DDM.LinearSystem;
using MGroup.Solvers.DDM.PSM.Dofs;
using MGroup.Solvers.DDM.PSM.InterfaceProblem;
using MGroup.Solvers.DDM.Tests.Commons;
using MGroup.Solvers.DDM.Tests.ExampleModels;
using MGroup.Solvers.DofOrdering;

namespace MGroup.Solvers.DDM.Tests.PSM
{
	public class PsmInterfaceProblemDofsTests
	{
		internal static void TestForLine1DInternal(IComputeEnvironment environment)
		{
			ComputeNodeTopology nodeTopology = Line1DExample.CreateNodeTopology();
			environment.Initialize(nodeTopology);

			IModel model = Line1DExample.CreateMultiSubdomainModel();
			DistributedOverlappingIndexer indexer = CreateDistributedOverlappingIndexer(environment, model);

			// Check
			Line1DExample.CheckDistributedIndexer(environment, nodeTopology, indexer);
		}

		internal static void TestForPlane2DInternal(IComputeEnvironment environment)
		{
			ComputeNodeTopology nodeTopology = Plane2DExample.CreateNodeTopology();
			environment.Initialize(nodeTopology);

			IModel model = Plane2DExample.CreateMultiSubdomainModel();
			DistributedOverlappingIndexer indexer = CreateDistributedOverlappingIndexer(environment, model);

			// Check
			Plane2DExample.CheckDistributedIndexer(environment, nodeTopology, indexer);
		}

		private static DistributedOverlappingIndexer CreateDistributedOverlappingIndexer(
			IComputeEnvironment environment, IModel model)
		{
			model.ConnectDataStructures();

			Dictionary<int, ISubdomainFreeDofOrdering> dofOrderings = environment.CalcNodeData(
				s => ModelUtilities.OrderDofs(model.GetSubdomain(s), model.AllDofs));
			var subdomainTopology = new SubdomainTopologyGeneral();
			subdomainTopology.Initialize(environment, model, s => dofOrderings[s]);

			Dictionary<int, MockSubdomainLinearSystem> linearSystems = environment.CalcNodeData(
				s => new MockSubdomainLinearSystem(s, dofOrderings[s]));
			Dictionary<int, PsmSubdomainDofs> subdomainDofs = environment.CalcNodeData(
				s => new PsmSubdomainDofs(model.GetSubdomain(s), linearSystems[s], true));

			subdomainTopology.FindCommonNodesBetweenSubdomains();
			subdomainTopology.FindCommonDofsBetweenSubdomains();
			environment.DoPerNode(s => subdomainDofs[s].SeparateFreeDofsIntoBoundaryAndInternal());
			return subdomainTopology.CreateDistributedVectorIndexer(s => subdomainDofs[s].DofOrderingBoundary);
		}

		private class MockSubdomainLinearSystem : ISubdomainLinearSystem
		{
			public MockSubdomainLinearSystem(int subdomainID, ISubdomainFreeDofOrdering dofOrdering)
			{
				this.SubdomainID = subdomainID;
				this.DofOrdering = dofOrdering;
			}

			public ISubdomainFreeDofOrdering DofOrdering { get; }

			public IMatrix Matrix => throw new NotImplementedException();

			public Vector RhsVector => throw new NotImplementedException();

			public Vector Solution { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

			public int SubdomainID { get; }
		}
	}
}

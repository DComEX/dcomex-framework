using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MGroup.Constitutive.Structural;
using MGroup.Constitutive.Structural.PlanarElements;
using MGroup.Environments;
using MGroup.FEM.Entities;
using MGroup.FEM.Structural.Elements;
using MGroup.MSolve.Constitutive;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.MSolve.Discretization.Loads;
using MGroup.MSolve.Meshes.Structured;
using MGroup.Solvers.DDM.FetiDP.Dofs;
using MGroup.Solvers.DDM.Partitioning;

//TODO: Allow option for prescribed displacement/load at corners or specific node index/ID.
namespace MGroup.Solvers.DDM.Tests.Commons
{
	public class UniformDdmModelBuilder2D
	{
		public enum BoundaryRegion
		{
			LeftSide, RightSide, UpperSide, LowerSide, UpperLeftCorner, UpperRightCorner, LowerLeftCorner, LowerRightCorner,
			Center
		}

		private List<(BoundaryRegion region, IDofType dof, double displacement)> prescribedDisplacements;
		private List<(BoundaryRegion region, IDofType dof, double load)> prescribedLoads;

		public UniformDdmModelBuilder2D()
		{
			prescribedDisplacements = new List<(BoundaryRegion region, IDofType dof, double displacement)>();
			prescribedLoads = new List<(BoundaryRegion region, IDofType dof, double load)>();
		}

		public double[] MinCoords { get; set; } = { -1.0, -1.0 };

		public double[] MaxCoords { get; set; } = { 1.0, 1.0 };

		public int[] NumElementsTotal { get; set; } = { 1, 1 };

		public int[] NumSubdomains { get; set; } = { 1, 1 };

		public int[] NumClusters { get; set; } = { 1, 1 };

		public double Thickness { get; set; } = 1.0;

		public IContinuumMaterial2D MaterialHomogeneous { get; set; }
			= new ElasticMaterial2D(StressState2D.PlaneStress) { YoungModulus = 1.0, PoissonRatio = 0.3 };

		public Func<int[], IContinuumMaterial2D> GetMaterialPerElementIndex { get; set; } = null;

		public (Model model, ComputeNodeTopology nodeTopology) BuildMultiSubdomainModel()
		{
			Model model = BuildSingleSubdomainModel();
			UniformCartesianMesh2D mesh = BuildMesh();
			var partitioner = new UniformMeshPartitioner2D(mesh, NumSubdomains, NumClusters);
			partitioner.Partition(model);
			ModelUtilities.DecomposeIntoSubdomains(model, partitioner.NumSubdomainsTotal, partitioner.GetSubdomainOfElement);

			var topology = new ComputeNodeTopology();
			for (int s = 0; s < partitioner.NumSubdomainsTotal; ++s)
			{
				topology.AddNode(s, partitioner.GetNeighboringSubdomains(s), partitioner.GetClusterOfSubdomain(s));
			}

			return (model, topology);
		}

		public Model BuildSingleSubdomainModel()
		{
			var model = new Model();
			model.SubdomainsDictionary[0] = new Subdomain(0);

			UniformCartesianMesh2D mesh = BuildMesh();

			// Nodes
			foreach ((int id, double[] coords) in mesh.EnumerateNodes())
			{
				model.NodesDictionary[id] = new Node(id, coords[0], coords[1]);
			}

			// Elements
			var dynamicProperties = new DynamicMaterial(1.0, 1.0, 1.0);
			var elemFactory = new ContinuumElement2DFactory(Thickness, MaterialHomogeneous, dynamicProperties);
			foreach ((int elementID, int[] nodeIDs) in mesh.EnumerateElements())
			{
				// Identify which material to use
				IContinuumMaterial2D elementMaterial;
				if (GetMaterialPerElementIndex == null)
				{
					elementMaterial = MaterialHomogeneous;
				}
				else
				{
					int[] elementIdx = mesh.GetElementIdx(elementID);
					elementMaterial = GetMaterialPerElementIndex(elementIdx);
				}

				Node[] nodes = nodeIDs.Select(n => model.NodesDictionary[n]).ToArray();
				var elementType = elemFactory.CreateElement(mesh.CellType, nodes, Thickness, elementMaterial, dynamicProperties);
				var element = new Element() { ID = elementID, ElementType = elementType };
				foreach (var node in nodes) element.AddNode(node);
				model.ElementsDictionary[element.ID] = element;
				model.SubdomainsDictionary[0].Elements.Add(element);
			}

			ApplyBoundaryConditions(model);

			return model;
		}

		/// <summary>
		/// If there are nodes belonging to <paramref name="region"/> taht are constrained along <paramref name="dof"/>, then 
		/// they will not be loaded, but the total <paramref name="load"/> will be divided by the total count of nodes, even the
		/// ones that will not be loaded.
		/// If there are multiple loads at the same (node, dof) then their sum will be used.
		/// </summary>
		/// <param name="load">Will be distributed evenly.</param>
		public void DistributeLoadAtNodes(BoundaryRegion region, IDofType dof, double load)
			=> prescribedLoads.Add((region, dof, load));

		public void PrescribeDisplacement(BoundaryRegion region, IDofType dof, double displacement)
			=> prescribedDisplacements.Add((region, dof, displacement));

		public static ICornerDofSelection FindCornerDofs(IModel model, int minCornerNodeMultiplicity = 2)
		{
			var cornerNodes = new HashSet<int>();
			foreach (ISubdomain subdomain in model.EnumerateSubdomains())
			{
				INode[] subdomainCorners = CornerNodeUtilities.FindCornersOfRectangle2D(subdomain);
				foreach (INode node in subdomainCorners)
				{
					if (node.Constraints.Count > 0) //TODO: allow only some dofs to be constrained
					{
						continue;
					}

					if (node.Subdomains.Count >= minCornerNodeMultiplicity)
					{
						cornerNodes.Add(node.ID);
					}
				}
			}

			var cornerDofs = new UserDefinedCornerDofSelection();
			foreach (int node in cornerNodes)
			{
				cornerDofs.AddCornerNode(node);
			}
			return cornerDofs;
		}

		private void ApplyBoundaryConditions(Model model)
		{
			double dx = (MaxCoords[0] - MinCoords[0]) / NumElementsTotal[0];
			double dy = (MaxCoords[1] - MinCoords[1]) / NumElementsTotal[1];
			double meshTolerance = 1E-10 * Math.Min(dx, dy);

			// Apply prescribed displacements
			foreach ((BoundaryRegion region, IDofType dof, double displacement) in prescribedDisplacements)
			{
				Node[] nodes = FindBoundaryNodes(region, model, meshTolerance);
				foreach (Node node in nodes)
				{
					// Search if there is already a prescribed displacement at this dof
					bool isFreeDof = true;
					foreach (Constraint constraint in node.Constraints)
					{
						if (constraint.DOF == dof)
						{
							if (constraint.Amount != displacement)
							{
								throw new Exception($"At node {node.ID}, dof = {dof}, " +
									$"both u = {displacement} and u = {constraint.Amount} have been prescribed.");
							}
							else
							{
								isFreeDof = false;

								// DO NOT break here. Let the loop iterate all constraints, since the expection above may be 
								// thrown for another constraint.
								//break; 
							}
						}
					}

					if (isFreeDof)
					{
						node.Constraints.Add(new Constraint() { DOF = dof, Amount = displacement });
					}
				}
			}

			// Apply prescribed loads
			foreach ((BoundaryRegion region, IDofType dof, double totalLoad) in prescribedLoads)
			{
				Node[] nodes = FindBoundaryNodes(region, model, meshTolerance);
				double load = totalLoad / nodes.Length;
				foreach (Node node in nodes)
				{
					// Search if there is already a prescribed displacement at this dof
					bool isFreeDof = true;
					foreach (Constraint constraint in node.Constraints)
					{
						if (constraint.DOF == dof)
						{
							isFreeDof = false;
							break;
						}
					}

					if (isFreeDof)
					{
						model.Loads.Add(new Load { Node = node, DOF = dof, Amount = load });
					}
				}
			}
		}

		private UniformCartesianMesh2D BuildMesh()
		{
			return new UniformCartesianMesh2D.Builder(MinCoords, MaxCoords, NumElementsTotal).SetMajorAxis(0).BuildMesh();
		}

		private Node[] FindBoundaryNodes(BoundaryRegion region, Model model, double tol)
		{
			double minX = MinCoords[0], minY = MinCoords[1], maxX = MaxCoords[0], maxY = MaxCoords[1]; // for brevity

			IEnumerable<Node> allNodes = model.NodesDictionary.Values;
			IEnumerable<Node> nodes;
			if (region == BoundaryRegion.LeftSide)
			{
				nodes = allNodes.Where(node => Math.Abs(node.X - minX) <= tol);
			}
			else if (region == BoundaryRegion.RightSide)
			{
				nodes = allNodes.Where(node => Math.Abs(node.X - maxX) <= tol);
			}
			else if (region == BoundaryRegion.LowerSide)
			{
				nodes = allNodes.Where(node => Math.Abs(node.Y - minY) <= tol);
			}
			else if (region == BoundaryRegion.UpperSide)
			{
				nodes = allNodes.Where(node => Math.Abs(node.Y - maxY) <= tol);
			}
			else if (region == BoundaryRegion.LowerLeftCorner)
			{
				nodes = allNodes.Where(node => (Math.Abs(node.X - minX) <= tol) && (Math.Abs(node.Y - minY) <= tol));
			}
			else if (region == BoundaryRegion.LowerRightCorner)
			{
				nodes = allNodes.Where(node => (Math.Abs(node.X - maxX) <= tol) && (Math.Abs(node.Y - minY) <= tol));
			}
			else if (region == BoundaryRegion.UpperLeftCorner)
			{
				nodes = allNodes.Where(node => (Math.Abs(node.X - minX) <= tol) && (Math.Abs(node.Y - maxY) <= tol));
			}
			else if (region == BoundaryRegion.UpperRightCorner)
			{
				nodes = allNodes.Where(node => (Math.Abs(node.X - maxX) <= tol) && (Math.Abs(node.Y - maxY) <= tol));
			}
			else if (region == BoundaryRegion.Center)
			{
				if ((NumElementsTotal[0] % 2 != 0) || (NumElementsTotal[1] % 2 != 0))
				{
					throw new ArgumentException(
						"To manipulate the node at the centre, the number of elements in each axis must be even");
				}

				double centerX = 0.5 * (MinCoords[0] + MaxCoords[0]);
				double centerY = 0.5 * (MinCoords[1] + MaxCoords[1]);
				nodes = allNodes.Where(node => (Math.Abs(node.X - centerX) <= tol) && (Math.Abs(node.Y - centerY) <= tol));
				Debug.Assert(nodes.Count() == 1);
			}
			else throw new Exception("Should not have reached this code");

			return nodes.ToArray();
		}
	}
}

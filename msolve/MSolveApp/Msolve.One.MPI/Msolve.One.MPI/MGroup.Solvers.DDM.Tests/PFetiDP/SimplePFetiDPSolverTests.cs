using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

using MGroup.Constitutive.Structural;
using MGroup.Constitutive.Thermal;
using MGroup.Environments;
using MGroup.LinearAlgebra.Distributed.IterativeMethods;
using MGroup.LinearAlgebra.Distributed.IterativeMethods.PCG;
using MGroup.LinearAlgebra.Distributed.IterativeMethods.PCG.Reorthogonalization;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Iterative.Termination;
using MGroup.LinearAlgebra.Matrices;
using MGroup.MSolve.DataStructures;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Solution;
using MGroup.NumericalAnalyzers;
using MGroup.Solvers.DDM.FetiDP.CoarseProblem;
using MGroup.Solvers.DDM.FetiDP.Dofs;
using MGroup.Solvers.DDM.FetiDP.StiffnessMatrices;
using MGroup.Solvers.DDM.LinearSystem;
using MGroup.Solvers.DDM.PFetiDP;
using MGroup.Solvers.DDM.Psm;
using MGroup.Solvers.DDM.PSM.InterfaceProblem;
using MGroup.Solvers.DDM.PSM.StiffnessMatrices;
using MGroup.Solvers.DDM.Tests.ExampleModels;
using MGroup.Solvers.DofOrdering;
using MGroup.Solvers.Results;

namespace MGroup.Solvers.DDM.Tests.PFetiDP
{
	public static class SimplePFetiDPSolverTests
	{
		internal static void TestForBrick3DInternal(IComputeEnvironment environment, bool isCoarseProblemDistributed,
			bool useCoarseJacobiPreconditioner, bool useReorthogonalizedPcg)
		{
			// Environment
			ComputeNodeTopology nodeTopology = Brick3DExample.CreateNodeTopology();
			environment.Initialize(nodeTopology);

			// Model
			IModel model = Brick3DExample.CreateMultiSubdomainModel();
			model.ConnectDataStructures(); //TODOMPI: this is also done in the analyzer
			ICornerDofSelection cornerDofs = Brick3DExample.GetCornerDofs(model);

			// Solver
			var solverFactory = new PFetiDPSolver<SymmetricCscMatrix>.Factory(
				environment, new PsmSubdomainMatrixManagerSymmetricCSparse.Factory(),
				cornerDofs, new FetiDPSubdomainMatrixManagerSymmetricCSparse.Factory(true));

			solverFactory.InterfaceProblemSolverFactory = new PsmInterfaceProblemSolverFactoryPcg()
			{
				MaxIterations = 200,
				ResidualTolerance = 1E-10
			};

			if (isCoarseProblemDistributed)
			{
				var coarseProblemFactory = new FetiDPCoarseProblemDistributed.Factory();
				if (useReorthogonalizedPcg)
				{
					var coarseProblemPcgBuilder = new ReorthogonalizedPcg.Builder();
					coarseProblemPcgBuilder.MaxIterationsProvider = new FixedMaxIterationsProvider(200);
					coarseProblemPcgBuilder.ResidualTolerance = 2E-12;
					coarseProblemPcgBuilder.DirectionVectorsRetention = new FixedDirectionVectorsRetention(40, true);
					coarseProblemFactory.CoarseProblemSolver = coarseProblemPcgBuilder.Build();
				}
				else
				{
					var coarseProblemPcgBuilder = new PcgAlgorithm.Builder();
					coarseProblemPcgBuilder.MaxIterationsProvider = new FixedMaxIterationsProvider(200);
					coarseProblemPcgBuilder.ResidualTolerance = 2E-12;
					coarseProblemFactory.CoarseProblemSolver = coarseProblemPcgBuilder.Build();
				}

				coarseProblemFactory.UseJacobiPreconditioner = useCoarseJacobiPreconditioner;
				solverFactory.CoarseProblemFactory = coarseProblemFactory;
			}
			else 
			{
				var coarseProblemMatrix = new FetiDPCoarseProblemMatrixSymmetricCSparse();
				solverFactory.CoarseProblemFactory = new FetiDPCoarseProblemGlobal.Factory(coarseProblemMatrix);
			}

			DistributedAlgebraicModel<SymmetricCscMatrix> algebraicModel = solverFactory.BuildAlgebraicModel(model);
			PsmSolver<SymmetricCscMatrix> solver = solverFactory.BuildSolver(model, algebraicModel);

			// Linear static analysis
			var problem = new ProblemStructural(model, algebraicModel, solver);
			var childAnalyzer = new LinearAnalyzer(model, algebraicModel, solver, problem);
			var parentAnalyzer = new StaticAnalyzer(model, algebraicModel, solver, problem, childAnalyzer);

			// Run the analysis
			parentAnalyzer.Initialize();
			parentAnalyzer.Solve();

			// Check results
			NodalResults expectedResults = Brick3DExample.GetExpectedNodalValues(model.AllDofs);
			double tolerance = 1E-7;
			environment.DoPerNode(subdomainID =>
			{
				NodalResults computedResults = algebraicModel.ExtractAllResults(subdomainID, solver.LinearSystem.Solution);
				if (!expectedResults.IsSuperSetOf(computedResults, tolerance, out string msg))
					Console.WriteLine("Wrong Results in SimplePFetiDPSolverTests.TestForBrick3D");
			});

			//Debug.WriteLine($"Num PCG iterations = {solver.PcgStats.NumIterationsRequired}," +
			//    $" final residual norm ratio = {solver.PcgStats.ResidualNormRatioEstimation}");

			// Check convergence
			int precision = 10;
			int pcgIterationsExpected = 29;
			double pcgResidualNormRatioExpected = 6.641424316172292E-11;
			IterativeStatistics stats = solver.InterfaceProblemSolutionStats;
			if(pcgIterationsExpected != stats.NumIterationsRequired || Math.Abs(pcgResidualNormRatioExpected - stats.ResidualNormRatioEstimation) > precision)
					Console.WriteLine("Wrong Results in SimplePFetiDPSolverTests.TestForBrick3D");
		}

		internal static void TestForPlane2DInternal(IComputeEnvironment environment, bool isCoarseProblemDistributed,
			bool useCoarseJacobiPreconditioner, bool useReorthogonalizedPcg)
		{
			// Environment
			ComputeNodeTopology nodeTopology = Plane2DExample.CreateNodeTopology();
			environment.Initialize(nodeTopology);

			// Model
			IModel model = Plane2DExample.CreateMultiSubdomainModel();
			model.ConnectDataStructures(); //TODOMPI: this is also done in the analyzer
			ICornerDofSelection cornerDofs = Plane2DExample.GetCornerDofs(model);

			// Solver
			var solverFactory = new PFetiDPSolver<SymmetricCscMatrix>.Factory(
				environment, new PsmSubdomainMatrixManagerSymmetricCSparse.Factory(),
				cornerDofs, new FetiDPSubdomainMatrixManagerSymmetricCSparse.Factory(true));

			solverFactory.InterfaceProblemSolverFactory = new PsmInterfaceProblemSolverFactoryPcg()
			{
				MaxIterations = 200,
				ResidualTolerance = 1E-10
			};

			if (isCoarseProblemDistributed)
			{
				var coarseProblemFactory = new FetiDPCoarseProblemDistributed.Factory();
				if (useReorthogonalizedPcg)
				{
					var coarseProblemPcgBuilder = new ReorthogonalizedPcg.Builder();
					coarseProblemPcgBuilder.DirectionVectorsRetention = new FixedDirectionVectorsRetention(30, true);
					coarseProblemPcgBuilder.MaxIterationsProvider = new FixedMaxIterationsProvider(200);
					coarseProblemPcgBuilder.ResidualTolerance = 2E-12;
					coarseProblemFactory.CoarseProblemSolver = coarseProblemPcgBuilder.Build();
				}
				else
				{
					var coarseProblemPcgBuilder = new PcgAlgorithm.Builder();
					coarseProblemPcgBuilder.MaxIterationsProvider = new FixedMaxIterationsProvider(200);
					coarseProblemPcgBuilder.ResidualTolerance = 2E-12;
					coarseProblemFactory.CoarseProblemSolver = coarseProblemPcgBuilder.Build();
				}
				
				coarseProblemFactory.UseJacobiPreconditioner = useCoarseJacobiPreconditioner;
				solverFactory.CoarseProblemFactory = coarseProblemFactory;
			}
			else 
			{
				var coarseProblemMatrix = new FetiDPCoarseProblemMatrixSymmetricCSparse();
				solverFactory.CoarseProblemFactory = new FetiDPCoarseProblemGlobal.Factory(coarseProblemMatrix);
			}

			DistributedAlgebraicModel<SymmetricCscMatrix> algebraicModel = solverFactory.BuildAlgebraicModel(model);
			PsmSolver<SymmetricCscMatrix> solver = solverFactory.BuildSolver(model, algebraicModel);

			// Linear static analysis
			var problem = new ProblemStructural(model, algebraicModel, solver);
			var childAnalyzer = new LinearAnalyzer(model, algebraicModel, solver, problem);
			var parentAnalyzer = new StaticAnalyzer(model, algebraicModel, solver, problem, childAnalyzer);

			// Run the analysis
			parentAnalyzer.Initialize();
			parentAnalyzer.Solve();

			// Check results
			NodalResults expectedResults = Plane2DExample.GetExpectedNodalValues(model.AllDofs);
			double tolerance = 1E-7;
			environment.DoPerNode(subdomainID =>
			{
				NodalResults computedResults = algebraicModel.ExtractAllResults(subdomainID, solver.LinearSystem.Solution);
				if(!expectedResults.IsSuperSetOf(computedResults, tolerance, out string msg))
					Console.WriteLine("Wrong Results in SimplePFetiDPSolverTests.TestForPlane2D");
			});

			//Debug.WriteLine($"Num PCG iterations = {solver.PcgStats.NumIterationsRequired}," +
			//    $" final residual norm ratio = {solver.PcgStats.ResidualNormRatioEstimation}");

			// Check convergence
			int precision = 10;
			int pcgIterationsExpected = 14;
			double pcgResidualNormRatioExpected = 2.868430313362798E-11;
			IterativeStatistics stats = solver.InterfaceProblemSolutionStats;
			if (pcgIterationsExpected != stats.NumIterationsRequired || Math.Abs(pcgResidualNormRatioExpected - stats.ResidualNormRatioEstimation) > precision)
				Console.WriteLine("Wrong Results in SimplePFetiDPSolverTests.TestForPlane2D");
		}
	}
}

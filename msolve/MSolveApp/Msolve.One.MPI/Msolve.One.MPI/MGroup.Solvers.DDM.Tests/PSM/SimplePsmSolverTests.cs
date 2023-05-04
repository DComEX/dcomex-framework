using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MGroup.Constitutive.Thermal;
using MGroup.Environments;
using MGroup.LinearAlgebra.Distributed.IterativeMethods;
using MGroup.LinearAlgebra.Iterative;
using MGroup.LinearAlgebra.Iterative.Termination;
using MGroup.LinearAlgebra.Matrices;
using MGroup.MSolve.DataStructures;
using MGroup.MSolve.Discretization;
using MGroup.NumericalAnalyzers;
using MGroup.Solvers.DDM.LinearSystem;
using MGroup.Solvers.DDM.Psm;
using MGroup.Solvers.DDM.PSM.Dofs;
using MGroup.Solvers.DDM.PSM.InterfaceProblem;
using MGroup.Solvers.DDM.PSM.StiffnessMatrices;
using MGroup.Solvers.DDM.Tests.ExampleModels;
using MGroup.Solvers.DofOrdering;
using MGroup.Solvers.Results;
//using TriangleNet;
namespace MGroup.Solvers.DDM.Tests.PSM
{
	public static class SimplePsmSolverTests
	{
		internal static void TestForBrick3DInternal(IComputeEnvironment environment)
		{
			// Environment
			ComputeNodeTopology nodeTopology = Brick3DExample.CreateNodeTopology();
			environment.Initialize(nodeTopology);

			// Model
			IModel model = Brick3DExample.CreateMultiSubdomainModel();
			model.ConnectDataStructures(); //TODOMPI: this is also done in the analyzer

			// Solver
			var solverFactory = new PsmSolver<SymmetricCscMatrix>.Factory(
				environment, new PsmSubdomainMatrixManagerSymmetricCSparse.Factory());
			solverFactory.InterfaceProblemSolverFactory = new PsmInterfaceProblemSolverFactoryPcg()
			{
				MaxIterations = 200,
				ResidualTolerance = 1E-10
			};
			DistributedAlgebraicModel<SymmetricCscMatrix> algebraicModel = solverFactory.BuildAlgebraicModel(model);
			PsmSolver<SymmetricCscMatrix> solver = solverFactory.BuildSolver(model, algebraicModel);

			// Linear static analysis
			var problem = new ProblemThermal(model, algebraicModel, solver);
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
					Console.WriteLine("Wrong results in SimplePsmSolverTests.TestForBrick3D");
			});

			//Debug.WriteLine($"Num PCG iterations = {solver.PcgStats.NumIterationsRequired}," +
			//    $" final residual norm ratio = {solver.PcgStats.ResidualNormRatioEstimation}");

			// Check convergence
			int precision = 10;
			int pcgIterationsExpected = 160;
			double pcgResidualNormRatioExpected = 7.487370033127084E-11;
			IterativeStatistics stats = solver.InterfaceProblemSolutionStats;
			if (pcgIterationsExpected != stats.NumIterationsRequired || Math.Abs(pcgResidualNormRatioExpected - stats.ResidualNormRatioEstimation) > precision)
				Console.WriteLine("Wrong results in SimplePsmSolverTests.TestForBrick3D");
		}

		internal static void TestForLine1DInternal(IComputeEnvironment environment)
		{
			// Environment
			ComputeNodeTopology nodeTopology = Line1DExample.CreateNodeTopology();
			environment.Initialize(nodeTopology);

			// Model
			IModel model = Line1DExample.CreateMultiSubdomainModel();
			model.ConnectDataStructures(); //TODOMPI: this is also done in the analyzer

			// Solver
			var solverFactory = new PsmSolver<SymmetricCscMatrix>.Factory(
				environment, new PsmSubdomainMatrixManagerSymmetricCSparse.Factory());
			DistributedAlgebraicModel<SymmetricCscMatrix> algebraicModel = solverFactory.BuildAlgebraicModel(model);
			PsmSolver<SymmetricCscMatrix> solver = solverFactory.BuildSolver(model, algebraicModel);

			// Linear static analysis
			var problem = new ProblemThermal(model, algebraicModel, solver);
			var childAnalyzer = new LinearAnalyzer(model, algebraicModel, solver, problem);
			var parentAnalyzer = new StaticAnalyzer(model, algebraicModel, solver, problem, childAnalyzer);

			// Run the analysis
			parentAnalyzer.Initialize();
			parentAnalyzer.Solve();

			// Check results
			NodalResults expectedResults = Line1DExample.GetExpectedNodalValues(model.AllDofs);
			double tolerance = 1E-7;
			environment.DoPerNode(subdomainID =>
			{
				NodalResults computedResults = algebraicModel.ExtractAllResults(subdomainID, solver.LinearSystem.Solution);
				if (!expectedResults.IsSuperSetOf(computedResults, tolerance, out string msg))
					Console.WriteLine("Wrong results in SimplePsmSolverTests.TestForLine1D");
			});

			//Debug.WriteLine($"Num PCG iterations = {solver.PcgStats.NumIterationsRequired}," +
			//    $" final residual norm ratio = {solver.PcgStats.ResidualNormRatioEstimation}");

			// Check convergence
			int precision = 10;
			int pcgIterationsExpected = 7;
			double pcgResidualNormRatioExpected = 0;
			IterativeStatistics stats = solver.InterfaceProblemSolutionStats;
			if (pcgIterationsExpected != stats.NumIterationsRequired || Math.Abs(pcgResidualNormRatioExpected - stats.ResidualNormRatioEstimation) > precision)
				Console.WriteLine("Wrong results in SimplePsmSolverTests.TestForLine1D");
		}

		internal static void TestForPlane2DInternal(IComputeEnvironment environment)
		{
			// Environment
			ComputeNodeTopology nodeTopology = Plane2DExample.CreateNodeTopology();
			environment.Initialize(nodeTopology);

			// Model
			IModel model = Plane2DExample.CreateMultiSubdomainModel();
			model.ConnectDataStructures(); //TODOMPI: this is also done in the analyzer

			// Solver
			var solverFactory = new PsmSolver<SymmetricCscMatrix>.Factory(
				environment, new PsmSubdomainMatrixManagerSymmetricCSparse.Factory());
			solverFactory.InterfaceProblemSolverFactory = new PsmInterfaceProblemSolverFactoryPcg()
			{
				MaxIterations = 200,
				ResidualTolerance = 1E-10
			};
			DistributedAlgebraicModel<SymmetricCscMatrix> algebraicModel = solverFactory.BuildAlgebraicModel(model);
			PsmSolver<SymmetricCscMatrix> solver = solverFactory.BuildSolver(model, algebraicModel);

			// Linear static analysis
			var problem = new ProblemThermal(model, algebraicModel, solver);
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
					Console.WriteLine("Wrong results in SimplePsmSolverTests.TestForPlane2D");
			});

			//Debug.WriteLine($"Num PCG iterations = {solver.PcgStats.NumIterationsRequired}," +
			//    $" final residual norm ratio = {solver.PcgStats.ResidualNormRatioEstimation}");

			// Check convergence
			int precision = 10;
			int pcgIterationsExpected = 63;
			double pcgResidualNormRatioExpected = 4.859075883397028E-11;
			IterativeStatistics stats = solver.InterfaceProblemSolutionStats;
			if (pcgIterationsExpected != stats.NumIterationsRequired || Math.Abs(pcgResidualNormRatioExpected - stats.ResidualNormRatioEstimation) > precision)
				Console.WriteLine("Wrong results in SimplePsmSolverTests.TestForPlane2D");
		}
	}
}

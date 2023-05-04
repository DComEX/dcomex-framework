using System;
using System.Collections.Generic;
using System.Net.Http.Headers;
using System.Text;
using MGroup.Environments.Mpi;
using MGroup.Solvers.DDM.Tests.PFetiDP;
using MGroup.Solvers.DDM.Tests.PSM;

//TODO: Allow client to register tests to be run, schedule them, run them with appropriate before and after messages, 
//      handle exceptions by adequately notifying user, but without crushing and omitting the next tests. 
//TODO: Hande the case of 1 machine running multiple tests that need different number of processes (make sure the max needed 
//      have been launched, deactivate the extras per test so that they do not mess with collectives). 
//      Handle the case of multiple machines running tests. Facilite user with command line parameters to MPI exec
namespace MGroup.Solvers.DDM.Tests
{
	public static class MpiTestSuite
	{
		public static void RunTestsWith5Processes()
		{
			IMpiGlobalOperationStrategy globalOperationStrategy = new MasterSlavesGlobalOperationStrategy(0);

			// Process 4 will be used only for global operations
			//IMpiGlobalOperationStrategy globalOperationStrategy = new MasterSlavesGlobalOperationStrategy(4);

			//IMpiGlobalOperationStrategy globalOperationStrategy = new DemocraticGlobalOperationStrategy();
			using (var mpiEnvironment = new MpiEnvironment(globalOperationStrategy))
			{
                // MpiDebugUtilities.AssistDebuggerAttachment();

				MpiDebugUtilities.DoSerially(MPI.Communicator.world,
					() => Console.WriteLine(
						$"Process {MPI.Communicator.world.Rank} on processor {MPI.Environment.ProcessorName}: " +
						$"Now running PsmInterfaceProblemDofsTests.TestForLine1DInternal"));
				PsmInterfaceProblemDofsTests.TestForLine1DInternal(mpiEnvironment);

				MpiDebugUtilities.DoSerially(MPI.Communicator.world,
					() => Console.WriteLine(
						$"Process {MPI.Communicator.world.Rank} on processor {MPI.Environment.ProcessorName}: " +
						$"Now running PsmInterfaceProblemDofsTests.TestForPlane2DInternal"));
				PsmInterfaceProblemDofsTests.TestForPlane2DInternal(mpiEnvironment);

				MpiDebugUtilities.DoSerially(MPI.Communicator.world,
					() => Console.WriteLine(
						$"Process {MPI.Communicator.world.Rank} on processor {MPI.Environment.ProcessorName}: " +
						$"Now running SimplePsmSolverTests.TestForLine1DInternal"));
				SimplePsmSolverTests.TestForLine1DInternal(mpiEnvironment);

				MpiDebugUtilities.DoSerially(MPI.Communicator.world,
					() => Console.WriteLine(
						$"Process {MPI.Communicator.world.Rank} on processor {MPI.Environment.ProcessorName}: " +
						$"Now running SimplePsmSolverTests.TestForPlane2DInternal"));
				SimplePsmSolverTests.TestForPlane2DInternal(mpiEnvironment);

				MpiDebugUtilities.DoSerially(MPI.Communicator.world,
					() => Console.WriteLine(
						$"Process {MPI.Communicator.world.Rank} on processor {MPI.Environment.ProcessorName}: " +
						$"Now running SimplePsmSolverTests.TestForBrick3DInternal"));
				SimplePsmSolverTests.TestForBrick3DInternal(mpiEnvironment);

				MpiDebugUtilities.DoSerially(MPI.Communicator.world,
					() => Console.WriteLine(
						$"Process {MPI.Communicator.world.Rank} on processor {MPI.Environment.ProcessorName}: " +
						$"Now running SimplePFetiDPSolverTests.TestForPlane2DInternal with distributed coarse problem."));
				SimplePFetiDPSolverTests.TestForPlane2DInternal(mpiEnvironment, true, false, false);

				MpiDebugUtilities.DoSerially(MPI.Communicator.world,
					() => Console.WriteLine(
						$"Process {MPI.Communicator.world.Rank} on processor {MPI.Environment.ProcessorName}: " +
						$"Now running SimplePFetiDPSolverTests.TestForBrick3DInternal with distributed coarse problem."));
				SimplePFetiDPSolverTests.TestForBrick3DInternal(mpiEnvironment, true, false, false);

				MpiDebugUtilities.DoSerially(MPI.Communicator.world,
					() => Console.WriteLine(
						$"Process {MPI.Communicator.world.Rank} on processor {MPI.Environment.ProcessorName}: " +
						$"Now running SimplePFetiDPSolverTests.TestForPlane2DInternal with global coarse problem."));
				SimplePFetiDPSolverTests.TestForPlane2DInternal(mpiEnvironment, false, false, false);

				MpiDebugUtilities.DoSerially(MPI.Communicator.world,
					() => Console.WriteLine(
						$"Process {MPI.Communicator.world.Rank} on processor {MPI.Environment.ProcessorName}: " +
						$"Now running SimplePFetiDPSolverTests.TestForBrick3DInternal with global coarse problem."));
				SimplePFetiDPSolverTests.TestForBrick3DInternal(mpiEnvironment, false, false, false);

				MpiDebugUtilities.DoSerially(MPI.Communicator.world,
					() => Console.WriteLine(
						$"Process {MPI.Communicator.world.Rank} on processor {MPI.Environment.ProcessorName}: " +
						$"All tests passed"));
			}
		}
	}
}

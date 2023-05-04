using System;
using MGroup.Solvers.DDM.Tests;
using Xunit;

namespace DotNetClient
{
    public class Program
    {
        [Fact]
        static void MPITest()
        {
            ILGPUTest.runILGPUTest();
            // Hello(args);
            // HostNamesTest(args);
            // CommunicationTest(args);
            MpiTestSuite.RunTestsWith5Processes();
            Assert.True(true);
		}

        static void Hello(string[] args)
        {
            MPI.Environment.Run(ref args, communicator =>
            {
                Console.WriteLine("Hello, from process number "
                                        + communicator.Rank + " of "
                                        + communicator.Size
                                        + " on processor: " + MPI.Environment.ProcessorName);
            });
        }

        static void CommunicationTest(string[] args)
        {
            MPI.Environment.Run(ref args, comm =>
            {
                if (comm.Size < 2)
                {
                    // Our ring needs at least two processes
                    Console.WriteLine("The Ring example must be run with at least two processes.");
                    Console.WriteLine("Try: mpiexec -np 4 ring.exe");
                }
                else if (comm.Rank == 0)
                {
                    // Rank 0 initiates communication around the ring
                    string data = "Hello, World!";

                    // Send "Hello, World!" to our right neighbor
                    comm.Send(data, (comm.Rank + 1) % comm.Size, 0);

                    // Receive data from our left neighbor
                    comm.Receive((comm.Rank + comm.Size - 1) % comm.Size, 0, out data);

                    // Add our own rank and write the results
                    data += " 0";
                    Console.WriteLine(data);
                }
                else
                {
                    // Receive data from our left neighbor
                    String data;
                    comm.Receive((comm.Rank + comm.Size - 1) % comm.Size, 0, out data);

                    // Add our own rank to the data
                    data = data + " " + comm.Rank.ToString() + ",";

                    // Pass on the intermediate to our right neighbor
                    comm.Send(data, (comm.Rank + 1) % comm.Size, 0);
                }
            });
        }

        static void HostNamesTest(string[] args)
        {
            MPI.Environment.Run(ref args, comm =>
            {
                string[] hostnames = comm.Gather(MPI.Environment.ProcessorName, 0);
                if (comm.Rank == 0)
                {
                    Array.Sort(hostnames);
                    foreach (string host in hostnames)
                        Console.WriteLine(host);
                }
            });

        }
    }
}

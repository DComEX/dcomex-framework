using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Algorithms;
using System;

namespace DotNetClient
{
    class ILGPUTest
    {
        static void PrintInformation()
        {
            using (var context = Context.CreateDefault())
            {
                foreach (var cpu in context.GetCPUDevices())
                {
                    cpu.PrintInformation();
                }

                foreach (var gpu in context.GetCudaDevices())
                {
                    gpu.PrintInformation();
                }
            }
        }

        static void PerformTestCUBLASOperation()
        {
            const int DataSize = 1024;

            using var context = Context.Create(builder => builder.Cuda().EnableAlgorithms());

            var cudaDevices = context.GetCudaDevices();

            if (cudaDevices.Count < 1)
            {
                Console.WriteLine("No CUDA devices found");
                return;
            }

            // Check for Cuda support
            foreach (var device in cudaDevices)
            {
                using var accelerator = device.CreateCudaAccelerator(context);
                Console.WriteLine($"Performing operations on {accelerator}");

                var buf = accelerator.Allocate1D<float>(DataSize);
                var buf2 = accelerator.Allocate1D<float>(DataSize);

                accelerator.Initialize(accelerator.DefaultStream, buf.View, 1.0f);
                accelerator.Initialize(accelerator.DefaultStream, buf2.View, 1.0f);

                // Initialize the CuBlas library using manual pointer mode handling
                // (default behavior)
                using (var blas = new CuBlas(accelerator))
                {
                    // Set pointer mode to Host to enable data transfer to CPU memory
                    blas.PointerMode = CuBlasPointerMode.Host;
                    float output = blas.Nrm2(buf.View.AsGeneral());

                    // Set pointer mode to Device to enable data transfer to GPU memory
                    blas.PointerMode = CuBlasPointerMode.Device;
                    blas.Nrm2(buf.View.AsGeneral(), buf2.View);

                    // Use pointer mode scopes to recover the previous pointer mode
                    using var scope = blas.BeginPointerScope(CuBlasPointerMode.Host);
                    float output2 = blas.Nrm2(buf.View.AsGeneral());
                }

                // Initialize the CuBlas<T> library using custom pointer mode handlers
                using (var blas = new CuBlas<CuBlasPointerModeHandlers.AutomaticMode>(accelerator))
                {
                    // Automatic transfer to host
                    float output = blas.Nrm2(buf.View.AsGeneral());

                    // Automatic transfer to device
                    blas.Nrm2(buf.View.AsGeneral(), buf2.View);
                }
            }
        }

        public static void runILGPUTest()
        {
            try
            {
                PrintInformation();
                Console.WriteLine("Print information operation completed succesfully");
            }
            catch (Exception e)
            {
                Console.WriteLine($"PrintInformation failed: {e.Message}\n\r{e.StackTrace}");
            }

            try
            {
                PerformTestCUBLASOperation();
                Console.WriteLine("CUBLAS operation completed succesfully");
            }
            catch (Exception e)
            {
                Console.WriteLine($"PerformTestCUBLASOperation failed: {e.Message}\n\r{e.StackTrace}");
            }
        }
    }
}

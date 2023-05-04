using ISAAR.MSolve.Analyzers;
using ISAAR.MSolve.Discretization;
using ISAAR.MSolve.Discretization.FreedomDegrees;
using ISAAR.MSolve.Discretization.Interfaces;
using ISAAR.MSolve.Discretization.Mesh;
using ISAAR.MSolve.FEM.Elements;
using ISAAR.MSolve.FEM.Entities;
using ISAAR.MSolve.FEM.Interfaces;
using ISAAR.MSolve.LinearAlgebra;
using ISAAR.MSolve.LinearAlgebra.Matrices;
using ISAAR.MSolve.Logging;
using ISAAR.MSolve.Materials;
using ISAAR.MSolve.Problems;
using ISAAR.MSolve.Solvers.Direct;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml;
using System.Xml.Linq;

namespace ISAAR.MSolve.MSolve4Korali
{
    enum ExitCode
    {
        Success = 0,
        InvalidArguments = -1,
        FileNotFound = -2,
        OutputFilenameEmpty = -3,
        DuplicateProcessIdentifer = -4,
        InvalidXML = -5,
        InvalidMesh = -6,
        InvalidPhysics = -7,
        InvalidParameters = -8,
        InvalidOutput = -9,
        ErrorWritingOutput = -10,
        AnalysisError = -11,
        UnknownError = -99
    }

    enum ProblemType
    {
        Unknown = -1,
        Thermal = 0,
        Cantilever,
        TumorGrowth,
    }

    class Program
    {
        #region Help message
        private const string helpMessage =
@"Executes MSolve as part of Korali software, solving a physics problem according to certain specifications.

Korali4MSolve inputfile outputfile_Name

  inputfile          Specifies the XML file with the parameters of the physics problem
  outputfile         Specifies the name and directory for the output XML file

";
        #endregion
        private static ProblemType problemType = ProblemType.Unknown;
        private static string inputFile = String.Empty;
        private static string outputFile = String.Empty;
        private static MeshParameters meshParameters;
        private static ThermalProblemParameters thermalProblemParameters;
        private static StructuralProblemParameters structuralProblemParameters;
        private static TumorProblemParameters tumorProblemParameters;
        private static Tuple<double, double>[] pointsOfInterest;
        private static double[] problemParameters = new double[0];

        /// <summary>
        /// Sets the input file and process identifier variables from the command line arguments
        /// </summary>
        /// <param name="args">A string list with the command line arguments</param>
        /// <returns>True if arguments are valid and false if not</returns>
        private static bool InitializeEnvironment(string[] args)
        {
            Console.WriteLine("InputFile: "+args[0].Trim());
            Console.WriteLine("OutputFile: "+args[1].Trim());
            string firstArgument = args.FirstOrDefault();
            if (args.Length != 2 || firstArgument == null)
            {
                Environment.ExitCode = (int)ExitCode.InvalidArguments;
                Console.WriteLine("Invalid Arguments!\n" + helpMessage);
                return false;
            }

            inputFile = args[0].Trim();
            outputFile = args[1].Trim();
            // inputFile = Path.Combine("/msolve", "ioDir", "model.xml");
            // processIdentifier = "1";
            if (File.Exists(inputFile) == false)
            {
                Environment.ExitCode = (int)ExitCode.FileNotFound;
                Console.WriteLine($"File {inputFile} does not exist.");
                return false;
            }

            if (String.IsNullOrEmpty(outputFile))
            {
                Environment.ExitCode = (int)ExitCode.OutputFilenameEmpty;
                Console.WriteLine(@"Output file name is empty.");
                return false;
            }

            return true;
        }

        /// <summary>
        /// Sets problem type and parameters from parsing the XML file from the inputFile path
        /// </summary>
        /// <returns>True if parsing was successful and false otherwise</returns>
        private static bool InitializeProblemParametersFromInputFile()
        {
            Console.WriteLine("Initializing problem parameters from input file...");
            try
            {
                Environment.ExitCode = (int)ExitCode.InvalidXML;
                var document = XDocument.Load(inputFile);

                Environment.ExitCode = (int)ExitCode.InvalidMesh;
                var meshElement = document.Root.Element("Mesh");

                Environment.ExitCode = (int)ExitCode.InvalidPhysics;
                var physicsElement = document.Root.Element("Physics");
                var problemTypeAsString = physicsElement.Attribute("type").Value.Trim();
                XElement parametersElement = null;
                Console.WriteLine("Initializing "+ problemTypeAsString + " problem...");
                switch (problemTypeAsString.ToUpper())
                {
                    case "THERMAL":
                        meshParameters = new MeshParameters()
                        {
                            ElementsX = Int32.Parse(meshElement.Element("ElementsX").Value.Trim(), CultureInfo.InvariantCulture),
                            ElementsY = Int32.Parse(meshElement.Element("ElementsY").Value.Trim(), CultureInfo.InvariantCulture),
                            LengthX = Double.Parse(meshElement.Element("LengthX").Value.Trim(), CultureInfo.InvariantCulture),
                            LengthY = Double.Parse(meshElement.Element("LengthY").Value.Trim(), CultureInfo.InvariantCulture),
                        };
                        problemType = ProblemType.Thermal;
                        thermalProblemParameters = new ThermalProblemParameters()
                        {
                            CommonThickness = Double.Parse(physicsElement.Element("CommonThickness").Value.Trim(), CultureInfo.InvariantCulture),
                            Conductivity = Double.Parse(physicsElement.Element("Conductivity").Value.Trim(), CultureInfo.InvariantCulture),
                            Density = Double.Parse(physicsElement.Element("Density").Value.Trim(), CultureInfo.InvariantCulture),
                            SpecialHeatCoefficient = Double.Parse(physicsElement.Element("SpecialHeatCoefficient").Value.Trim(), CultureInfo.InvariantCulture),
                            TemperatureAtBoundaries = Double.Parse(physicsElement.Element("TemperatureAtBoundaries").Value.Trim(), CultureInfo.InvariantCulture),
                            HeatSourceMagnitude = Double.Parse(physicsElement.Element("HeatSourceMagnitude").Value.Trim(), CultureInfo.InvariantCulture),
                            HeatSourceSpread = Double.Parse(physicsElement.Element("HeatSourceSpread").Value.Trim(), CultureInfo.InvariantCulture),
                        };

                        Environment.ExitCode = (int)ExitCode.InvalidParameters;
                        parametersElement = document.Root.Element("Parameters");
                        thermalProblemParameters.Theta1 = Double.Parse(parametersElement.Element("Theta1").Value.Trim(), CultureInfo.InvariantCulture);
                        thermalProblemParameters.Theta2 = Double.Parse(parametersElement.Element("Theta2").Value.Trim(), CultureInfo.InvariantCulture);

                        Environment.ExitCode = (int)ExitCode.InvalidOutput;
                        pointsOfInterest = document.Root.Element("Output").Elements("Temperature")
                            .Select(x => new Tuple<double, double>
                            (
                                Double.Parse(x.Attribute("X").Value, CultureInfo.InvariantCulture),
                                Double.Parse(x.Attribute("Y").Value, CultureInfo.InvariantCulture)
                            ))
                            .ToArray();
                        break;
                    case "CANTILEVER":
                        meshParameters = new MeshParameters()
                        {
                            ElementsX = Int32.Parse(meshElement.Element("ElementsX").Value.Trim(), CultureInfo.InvariantCulture),
                            ElementsY = Int32.Parse(meshElement.Element("ElementsY").Value.Trim(), CultureInfo.InvariantCulture),
                            LengthX = Double.Parse(meshElement.Element("LengthX").Value.Trim(), CultureInfo.InvariantCulture),
                            LengthY = Double.Parse(meshElement.Element("LengthY").Value.Trim(), CultureInfo.InvariantCulture),
                        };
                        problemType = ProblemType.Cantilever;
                        structuralProblemParameters = new StructuralProblemParameters()
                        {
                            CommonThickness = Double.Parse(physicsElement.Element("CommonThickness").Value.Trim(), CultureInfo.InvariantCulture),
                            YoungModulus = Double.Parse(physicsElement.Element("YoungModulus").Value.Trim(), CultureInfo.InvariantCulture),
                            PoissonRatio = Double.Parse(physicsElement.Element("PoissonRatio").Value.Trim(), CultureInfo.InvariantCulture),
                            Density = Double.Parse(physicsElement.Element("Density").Value.Trim(), CultureInfo.InvariantCulture),
                            RayleighMassCoefficient = Double.Parse(physicsElement.Element("RayleighMassCoefficient").Value.Trim(), CultureInfo.InvariantCulture),
                            RayleighStiffnessCoefficient = Double.Parse(physicsElement.Element("RayleighStiffnessCoefficient").Value.Trim(), CultureInfo.InvariantCulture),
                            DisplacementXAtBoundaries = Double.Parse(physicsElement.Element("DisplacementXAtBoundaries").Value.Trim(), CultureInfo.InvariantCulture),
                            DisplacementYAtBoundaries = Double.Parse(physicsElement.Element("DisplacementYAtBoundaries").Value.Trim(), CultureInfo.InvariantCulture),
                            LoadMagnitudeX = Double.Parse(physicsElement.Element("LoadMagnitudeX").Value.Trim(), CultureInfo.InvariantCulture),
                            LoadMagnitudeY = Double.Parse(physicsElement.Element("LoadMagnitudeY").Value.Trim(), CultureInfo.InvariantCulture),
                            LoadSpread = Double.Parse(physicsElement.Element("LoadSpread").Value.Trim(), CultureInfo.InvariantCulture),
                        };

                        Environment.ExitCode = (int)ExitCode.InvalidParameters;
                        parametersElement = document.Root.Element("Parameters");
                        structuralProblemParameters.Theta1 = Double.Parse(parametersElement.Element("Theta1").Value.Trim(), CultureInfo.InvariantCulture);
                        structuralProblemParameters.Theta2 = Double.Parse(parametersElement.Element("Theta2").Value.Trim(), CultureInfo.InvariantCulture);

                        Environment.ExitCode = (int)ExitCode.InvalidOutput;
                        pointsOfInterest = document.Root.Element("Output").Elements("Displacement")
                            .Select(x => new Tuple<double, double>
                            (
                                Double.Parse(x.Attribute("X").Value, CultureInfo.InvariantCulture),
                                Double.Parse(x.Attribute("Y").Value, CultureInfo.InvariantCulture)
                            ))
                            .ToArray();
                        break;
                    case "TUMORGROWTH":
                        problemType = ProblemType.TumorGrowth;
                        tumorProblemParameters = new TumorProblemParameters()
                        {
                            fileName= meshElement.Element("File").Value.Trim(),
                            kappaNormal= 0,
                            miTumor= 0,
                            timeStep= Double.Parse(physicsElement.Element("Timestep").Value.Trim(), CultureInfo.InvariantCulture),
                            totalTime= Double.Parse(physicsElement.Element("Time").Value.Trim(), CultureInfo.InvariantCulture)
                        };

                        Environment.ExitCode = (int)ExitCode.InvalidParameters;
                        parametersElement = document.Root.Element("Parameters");
                        tumorProblemParameters.kappaNormal = Double.Parse(parametersElement.Element("k1").Value.Trim(), CultureInfo.InvariantCulture);
                        tumorProblemParameters.miTumor = Double.Parse(parametersElement.Element("mu").Value.Trim(), CultureInfo.InvariantCulture);

                        Environment.ExitCode = (int)ExitCode.InvalidOutput;
                        break;
                    default:
                        Console.WriteLine($"Problem type '{problemTypeAsString}' is not recognized.");
                        return false;
                }

                Console.WriteLine("Success initializing problem");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"XML parsing of {inputFile} failed.\n\r{Enum.GetName(typeof(ExitCode), Environment.ExitCode)}\n\r{ex.Message}");
                return false;
            }
        }

        private static void BuildRectangularMesh(Model model, Func<CellType, IReadOnlyList<Node>, IFiniteElement> elementFactory)
        {
            Console.WriteLine("Building Rectangular Mesh...");
            double stepX = meshParameters.LengthX / meshParameters.ElementsX;
            double stepY = meshParameters.LengthY / meshParameters.ElementsY;
            int nodesX = meshParameters.ElementsX + 1;
            int nodesY = meshParameters.ElementsY + 1;

            var nodes = Enumerable.Range(0, nodesY)
                .Select(y => Enumerable.Range(0, nodesX)
                    .Select(x => new Node(nodesX * y + x + 1, x * stepX, y * stepY)))
                .SelectMany(x => x);
            foreach (var node in nodes)
            {
                model.NodesDictionary.Add(node.ID, node);
            }

            var elementNodePairs = Enumerable.Range(0, meshParameters.ElementsY)
                .Select(y => Enumerable.Range(0, meshParameters.ElementsX)
                    .Select(x => new Tuple<int, Node[]>(meshParameters.ElementsX * y + x + 1, new[]
                    {
                        model.Nodes[y * nodesX + x],
                        model.Nodes[y * nodesX + x + 1],
                        model.Nodes[(y + 1) * nodesX + x + 1],
                        model.Nodes[(y + 1) * nodesX + x]
                    })))
                .SelectMany(x => x);
            foreach (var en in elementNodePairs)
            {
                var element = new Element() { ID = en.Item1, ElementType = elementFactory(CellType.Quad4, en.Item2) };
                model.ElementsDictionary.Add(en.Item1, element);
                model.SubdomainsDictionary[0].Elements.Add(element);
                foreach (var node in en.Item2)
                {
                    element.NodesDictionary.Add(node.ID, node);
                }
            }
        }

        private static bool WriteOutputToXml(Dictionary<Tuple<double, double>, double> outputValues, string physicalQuantity)
        {
            Console.WriteLine("Writing output to xml...");
            Environment.ExitCode = (int)ExitCode.ErrorWritingOutput;
            // string outputFileName = $"MSolveOutput-{processIdentifier}.xml";
            // string outputFile = Path.Combine("/msolve", "ioDir", outputFileName);
            var settings = new XmlWriterSettings
            {
                Indent = true,
                IndentChars = "\t",
                NewLineChars = "\r\n",
                NewLineHandling = NewLineHandling.Replace
            };

            try
            {
                using (var xmlWriter = XmlWriter.Create(outputFile, settings))
                {
                    xmlWriter.WriteStartDocument();

                    xmlWriter.WriteStartElement("MSolve4Korali_output");
                    xmlWriter.WriteAttributeString("version", "1.0");
                    xmlWriter.WriteStartElement(physicalQuantity + "s");

                    foreach (var v in outputValues)
                    {
                        xmlWriter.WriteStartElement(physicalQuantity);
                        xmlWriter.WriteAttributeString("X", v.Key.Item1.ToString("0.00"));
                        xmlWriter.WriteAttributeString("Y", v.Key.Item2.ToString("0.00"));
                        xmlWriter.WriteString(v.Value.ToString());
                        xmlWriter.WriteEndElement();
                    }

                    xmlWriter.WriteEndElement();
                    xmlWriter.WriteEndElement();
                    xmlWriter.WriteEndDocument();

                    xmlWriter.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"XML writing of {outputFile} failed.\n\r{Enum.GetName(typeof(ExitCode), Environment.ExitCode)}\n\r{ex.Message}");
                return false;
            }
            Console.WriteLine("Success writing output to xml");

            return true;
        }
        private static bool WriteOutputToXml(double outputValue, double timesteps, bool runSuccess, string physicalQuantity)
        {
            Console.WriteLine("Writing output to xml...");
            Environment.ExitCode = (int)ExitCode.ErrorWritingOutput;
            var settings = new XmlWriterSettings
            {
                Indent = true,
                IndentChars = "\t",
                NewLineChars = "\r\n",
                NewLineHandling = NewLineHandling.Replace
            };

            try
            {
                using (var xmlWriter = XmlWriter.Create(outputFile, settings))
                {
                    xmlWriter.WriteStartDocument();

                    xmlWriter.WriteStartElement("MSolve4Korali_output");
                    xmlWriter.WriteAttributeString("version", "1.0");

                    xmlWriter.WriteElementString(physicalQuantity, outputValue.ToString());
                    xmlWriter.WriteElementString("Timeteps", timesteps.ToString());
                    xmlWriter.WriteElementString("SolutionMsg", runSuccess ? "Success" : "Fail");

                    xmlWriter.WriteEndElement();
                    xmlWriter.WriteEndDocument();

                    xmlWriter.Close();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"XML writing of {outputFile} failed.\n\r{Enum.GetName(typeof(ExitCode), Environment.ExitCode)}\n\r{ex.Message}");
                return false;
            }
            Console.WriteLine("Success writing output to xml");

            return true;
        }

        private static bool SolveThermalProblem()
        {
            Console.WriteLine("Solving problem...");
            var model = new Model();
            model.SubdomainsDictionary.Add(0, new Subdomain(0));

            var elementFactory = new ThermalElement2DFactory(thermalProblemParameters.CommonThickness,
                new ThermalMaterial(thermalProblemParameters.Density, thermalProblemParameters.SpecialHeatCoefficient, thermalProblemParameters.Conductivity));
            BuildRectangularMesh(model, elementFactory.CreateElement);
            var constrainedNodes = model.Nodes.Where(x => x.X1 == 0 || x.X1 == meshParameters.LengthX || x.X2 == 0 || x.X2 == meshParameters.LengthY);
            foreach (var node in constrainedNodes)
            {
                node.Constraints.Add(new Constraint() { DOF = ThermalDof.Temperature, Amount = thermalProblemParameters.TemperatureAtBoundaries });
            }

            var dx = meshParameters.LengthX / meshParameters.ElementsX;
            var dy = meshParameters.LengthY / meshParameters.ElementsY;
            foreach (var node in model.Nodes.Except(constrainedNodes))
            {
                model.Loads.Add(new Load()
                {
                    Amount = thermalProblemParameters.HeatSourceMagnitude * Math.Exp(-(Math.Pow(node.X1 - thermalProblemParameters.Theta1, 2) + Math.Pow(node.X2 - thermalProblemParameters.Theta2, 2)) / thermalProblemParameters.HeatSourceSpread) * dx * dy,
                    Node = node,
                    DOF = ThermalDof.Temperature
                });
            }

            SkylineSolver solver = new SkylineSolver.Builder().BuildSolver(model);
            var provider = new ProblemThermal(model, solver);
            var childAnalyzer = new LinearAnalyzer(model, solver, provider);
            var parentAnalyzer = new StaticAnalyzer(model, solver, provider, childAnalyzer);
            parentAnalyzer.Initialize();
            parentAnalyzer.Solve();

            var outputValues = new Dictionary<Tuple<double, double>, double>();
            foreach (var n in pointsOfInterest)
            {
                var nearestNode = model.Nodes.Select(x => new { Node = x, Distance = Math.Sqrt(Math.Pow(n.Item1 - x.X1, 2) + Math.Pow(n.Item2 - x.X2, 2)) }).OrderBy(x => x.Distance).First().Node;
                var dof = 0;
                model.GlobalDofOrdering.GlobalFreeDofs.TryGetValue(nearestNode, ThermalDof.Temperature, out dof);
                outputValues.Add(n, dof == 0 ? 0 : solver.LinearSystems[0].Solution[dof]);
            }

            return WriteOutputToXml(outputValues, "Temperature");
        }

        private static bool SolveCantileverProblem()
        {
            Console.WriteLine("Solving problem...");

            var model = new Model();
            model.SubdomainsDictionary.Add(0, new Subdomain(0));

            var elementFactory = new ContinuumElement2DFactory(structuralProblemParameters.CommonThickness,
                new ElasticMaterial2D(StressState2D.PlaneStress) { YoungModulus = structuralProblemParameters.YoungModulus, PoissonRatio = structuralProblemParameters.PoissonRatio },
                new DynamicMaterial(structuralProblemParameters.Density, structuralProblemParameters.RayleighMassCoefficient, structuralProblemParameters.RayleighStiffnessCoefficient));
            BuildRectangularMesh(model, elementFactory.CreateElement);

            SkylineSolver solver = new SkylineSolver.Builder().BuildSolver(model);
            var provider = new ProblemStructural(model, solver);
            var childAnalyzer = new LinearAnalyzer(model, solver, provider);
            var parentAnalyzer = new StaticAnalyzer(model, solver, provider, childAnalyzer);
            parentAnalyzer.Initialize();
            var array = solver.LinearSystems[0].Matrix.CopytoArray2D();
            var lines = new string[18];
            for (int i = 0; i < 18; i++)
            {
                var line = String.Empty;
                for (int j = 0; j < 18; j++)
                {
                    line += array[i, j] + ";";
                }
                lines[i] = line;
            }
            File.WriteAllLines(Path.Combine("/msolve", "ioDir", "cantileverHalf.txt"), lines);
            parentAnalyzer.Solve();

            var outputValues = new Dictionary<Tuple<double, double>, double>();
            foreach (var n in pointsOfInterest)
            {
                var nearestNode = model.Nodes.Select(x => new { Node = x, Distance = Math.Sqrt(Math.Pow(n.Item1 - x.X1, 2) + Math.Pow(n.Item2 - x.X2, 2)) }).OrderBy(x => x.Distance).First().Node;
                var dof = 0;
                model.GlobalDofOrdering.GlobalFreeDofs.TryGetValue(nearestNode, StructuralDof.TranslationX, out dof);
                outputValues.Add(n, dof == 0 ? 0 : solver.LinearSystems[0].Solution[dof]);
            }

            return WriteOutputToXml(outputValues, "Displacement X");
        }

        private static bool SolveTumorGrowthProblem()
        {
			var equationModel = new MonophasicEquationModel(tumorProblemParameters);
            var u1X = new double[(int)(tumorProblemParameters.totalTime / tumorProblemParameters.timeStep)];
            var u1Y = new double[(int)(tumorProblemParameters.totalTime / tumorProblemParameters.timeStep)];
            var u1Z = new double[(int)(tumorProblemParameters.totalTime / tumorProblemParameters.timeStep)];

            Dictionary<double, double[]> Solution = new Dictionary<double, double[]>();

            var staggeredAnalyzer = new MGroup.NumericalAnalyzers.Staggered.StepwiseStaggeredAnalyzer(equationModel.ParentAnalyzers, equationModel.ParentSolvers, equationModel.CreateModel, maxStaggeredSteps: 3, tolerance: 1e-5);
            try
            {
                Environment.ExitCode = (int)ExitCode.InvalidParameters;
                for (int currentTimeStep = 0; currentTimeStep < tumorProblemParameters.totalTime / tumorProblemParameters.timeStep; currentTimeStep++)
                {
                    equationModel.CurrentTimeStep = currentTimeStep;
                    equationModel.CreateModel(equationModel.ParentAnalyzers, equationModel.ParentSolvers);
                    staggeredAnalyzer.SolveCurrentStep();

                    var allValues = ((MGroup.NumericalAnalyzers.Logging.DOFSLog)equationModel.ParentAnalyzers[0].ChildAnalyzer.Logs[0]).DOFValues.Select(x => x.val).ToArray();

                    u1X[currentTimeStep] = allValues[0];
                    u1Y[currentTimeStep] = allValues[1];
                    u1Z[currentTimeStep] = allValues[2];

                    if (Solution.ContainsKey(currentTimeStep))
                    {
                        Solution[currentTimeStep] = allValues;
                        Console.WriteLine($"Time step: {currentTimeStep}");
                        Console.WriteLine($"Displacement vector: {string.Join(", ", Solution[currentTimeStep])}");
                    }
                    else
                    {
                        Solution.Add(currentTimeStep, allValues);
                    }

                    for (int j = 0; j < equationModel.ParentAnalyzers.Length; j++)
                    {
                        (equationModel.ParentAnalyzers[j] as MGroup.NumericalAnalyzers.Dynamic.PseudoTransientAnalyzer).AdvanceStep();
                    }

                    for (int j = 0; j < equationModel.ParentAnalyzers.Length; j++)
                    {
                        equationModel.AnalyzerStates[j] = equationModel.ParentAnalyzers[j].CreateState();
                        equationModel.NLAnalyzerStates[j] = equationModel.NLAnalyzers[j].CreateState();
                    }


                    Console.WriteLine($"Displacement vector: {string.Join(", ", Solution[currentTimeStep])}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Analysis failed on step {equationModel.CurrentTimeStep}.\n\r{Enum.GetName(typeof(ExitCode), Environment.ExitCode)}\n\r{ex.Message}");
                return WriteOutputToXml(Solution[equationModel.CurrentTimeStep-1][0], equationModel.CurrentTimeStep-1, false, "Volume");
            }

            return WriteOutputToXml(Solution[equationModel.CurrentTimeStep][0], equationModel.CurrentTimeStep, true, "Volume");
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Starting MSolve4Korali");
            Environment.ExitCode = (int)ExitCode.UnknownError;
            if (InitializeEnvironment(args) == false)
            {
                Console.WriteLine("Could not initialize Environment. Exiting.");
                return;
            }
            if (InitializeProblemParametersFromInputFile() == false)
            {
                Console.WriteLine("Could not initialize problem parameters from input. Exiting.");
                return;
            }
            switch (problemType)
            {
                case ProblemType.Cantilever:
                    if (SolveCantileverProblem() == false)
                    {
                        Console.WriteLine("Could not solve Cantilever problem. Exiting.");
                        return;
                    }
                    break;
                case ProblemType.Thermal:
                    if (SolveThermalProblem() == false)
                    {
                        Console.WriteLine("Could not solve Thermal problem. Exiting.");
                        return;
                    }
                    break;
                case ProblemType.TumorGrowth:
                    if (SolveTumorGrowthProblem() == false)
                    {
                        Console.WriteLine("Could not solve Thermal problem. Exiting.");
                        return;
                    }
                    break;
                default:
                    Console.WriteLine("Could not resolve problem type. Exiting.");
                    return;
            }

            Environment.ExitCode = (int)ExitCode.Success;
            Console.WriteLine("Programm exited with code " + Environment.ExitCode);
        }
    }
}


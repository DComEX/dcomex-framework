namespace ISAAR.MSolve.MSolve4Korali
{
    public class StructuralProblemParameters
    {
        public double CommonThickness { get; set; }
        public double Density { get; set; }
        public double YoungModulus { get; set; }
        public double PoissonRatio { get; set; }
        public double RayleighStiffnessCoefficient { get; set; }
        public double RayleighMassCoefficient { get; set; }
        public double DisplacementXAtBoundaries { get; set; }
        public double DisplacementYAtBoundaries { get; set; }
        public double LoadMagnitudeX { get; set; }
        public double LoadMagnitudeY { get; set; }
        public double LoadSpread { get; set; }
        public double Theta1 { get; set; }
        public double Theta2 { get; set; }
    }
}

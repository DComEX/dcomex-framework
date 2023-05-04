namespace ISAAR.MSolve.MSolve4Korali
{
    public class ThermalProblemParameters
    {
        public double CommonThickness { get; set; }
        public double Density { get; set; }
        public double SpecialHeatCoefficient { get; set; }
        public double Conductivity { get; set; }
        public double TemperatureAtBoundaries { get; set; }
        public double HeatSourceMagnitude { get; set; }
        public double HeatSourceSpread { get; set; }
        public double Theta1 { get; set; }
        public double Theta2 { get; set; }
    }
}

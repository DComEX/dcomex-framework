using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MGroup.Environments;
using MGroup.Environments.Mpi;
using MGroup.MSolve.Solution.LinearSystem;
using MGroup.MSolve.DataStructures;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Solution.AlgebraicModel;
using MGroup.Solvers.DofOrdering;

namespace MGroup.Solvers.DDM.Tests.Commons
{
	public static class Utilities
	{
		public static bool AreEqual(int[] expected, int[] computed)
		{
			if (expected.Length != computed.Length)
			{
				return false;
			}
			for (int i = 0; i < expected.Length; ++i)
			{
				if (expected[i] != computed[i])
						{
					return false;
				}
			}
			return true;
		}
	}
}

using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.Environments.Tests
{
    internal static class Utilities
    {
        internal static bool AreEqual(int[] expected, int[] computed)
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

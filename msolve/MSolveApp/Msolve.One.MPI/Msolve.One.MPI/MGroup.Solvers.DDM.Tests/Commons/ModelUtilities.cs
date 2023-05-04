using System;
using System.Collections.Generic;
using System.Text;
using MGroup.FEM.Entities;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.Solvers.DofOrdering;

namespace MGroup.Solvers.DDM.Tests.Commons
{
	public static class ModelUtilities
	{
		public static ISubdomainFreeDofOrdering OrderDofs(ISubdomain subdomain, ActiveDofs allDofs)
		{
			var dofOrderer = new NodeMajorDofOrderingStrategy();
			(int numSubdomainFreeDofs, IntDofTable subdomainFreeDofs) = dofOrderer.OrderSubdomainDofs(subdomain, allDofs);
			var dofOrdering = new SubdomainFreeDofOrderingCaching(numSubdomainFreeDofs, subdomainFreeDofs, allDofs);
			return dofOrdering;
		}

		public static void DecomposeIntoSubdomains(this Model model, int numSubdomains, Func<int, int> getSubdomainOfElement)
		{
			model.SubdomainsDictionary.Clear();
			foreach (Node node in model.NodesDictionary.Values) node.Subdomains.Clear();
			foreach (Element element in model.ElementsDictionary.Values) element.SubdomainID = int.MinValue;

			for (int s = 0; s < numSubdomains; ++s)
			{
				model.SubdomainsDictionary[s] = new Subdomain(s);
			}
			foreach (Element element in model.ElementsDictionary.Values)
			{
				Subdomain subdomain = model.SubdomainsDictionary[getSubdomainOfElement(element.ID)];
				subdomain.Elements.Add(element);
			}

			model.ConnectDataStructures();
		}
	}
}

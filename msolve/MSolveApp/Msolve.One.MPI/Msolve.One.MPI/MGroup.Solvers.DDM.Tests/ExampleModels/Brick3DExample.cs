using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MGroup.Constitutive.Structural;
using MGroup.Constitutive.Structural.ContinuumElements;
using MGroup.Environments;
using MGroup.FEM.Entities;
using MGroup.FEM.Structural.Elements;
using MGroup.MSolve.DataStructures;
using MGroup.MSolve.Discretization;
using MGroup.MSolve.Discretization.Dofs;
using MGroup.MSolve.Discretization.Loads;
using MGroup.MSolve.Discretization.Meshes.Structured;
using MGroup.Solvers.DDM.FetiDP.Dofs;
using MGroup.Solvers.DDM.Tests.Commons;
using MGroup.Solvers.Results;

//TODO: different number of clusters, subdomains, elements per axis. Try to make this as nonsymmetric as possible, 
//      but keep subdomain-elements ratio constant to have the same stiffnesses.
//TODO: Finding the correct indexing data by hand is going to be very difficult. Take them from a correct solution and 
//      hardcode them.
namespace MGroup.Solvers.DDM.Tests.ExampleModels
{
	public class Brick3DExample
	{
		private const double E = 1.0, v = 0.3;
		private const double load = 100;

		public static double[] MinCoords => new double[] { 0, 0, 0 };

		public static double[] MaxCoords => new double[] { 6, 9, 12 };

		public static int[] NumElements => new int[] { 4, 6, 8 };

		public static int[] NumSubdomains => new int[] { 2, 3, 4 };

		public static int[] NumClusters => new int[] { 2, 1, 2 };

		public static ComputeNodeTopology CreateNodeTopology()
		{
			var nodeTopology = new ComputeNodeTopology();
			Dictionary<int, int> clustersOfSubdomains = GetSubdomainClusters();
			Dictionary<int, int[]> neighborsOfSubdomains = GetSubdomainNeighbors();
			for (int s = 0; s < NumSubdomains[0] * NumSubdomains[1] * NumSubdomains[2]; ++s)
			{
				nodeTopology.AddNode(s, neighborsOfSubdomains[s], clustersOfSubdomains[s]);
			}
			return nodeTopology;
		}

		public static Model CreateSingleSubdomainModel()
		{
			var builder = new UniformDdmModelBuilder3D();
			builder.MinCoords = MinCoords;
			builder.MaxCoords = MaxCoords;
			builder.NumElementsTotal = NumElements;
			builder.NumSubdomains = NumSubdomains;
			builder.NumClusters = NumClusters;
			builder.MaterialHomogeneous = new ElasticMaterial3D() { YoungModulus = E, PoissonRatio = v };
			Model model = builder.BuildSingleSubdomainModel();

			// Boundary conditions
			//TODO: hardcode the node IDs
			var constrainedNodes = new List<int>();
			constrainedNodes.Add(/*mesh.GetNodeID(new int[] { 0, 0, 0 })*/0);
			constrainedNodes.Add(/*mesh.GetNodeID(new int[] { mesh.NumNodes[0] - 1, 0, 0 })/*6*/4);
			constrainedNodes.Add(/*mesh.GetNodeID(new int[] { 0, mesh.NumNodes[1] - 1, 0 })*//*63*/30);
			foreach (int nodeID in constrainedNodes)
			{
				Node node = model.NodesDictionary[nodeID];
				node.Constraints.Add(new Constraint() { DOF = StructuralDof.TranslationX, Amount = 0 });
				node.Constraints.Add(new Constraint() { DOF = StructuralDof.TranslationY, Amount = 0 });
				node.Constraints.Add(new Constraint() { DOF = StructuralDof.TranslationZ, Amount = 0 });
			}

			var loadedNodes = new List<int>();
			loadedNodes.Add(/*mesh.GetNodeID(new int[] { mesh.NumNodes[0] - 1, mesh.NumNodes[1] - 1, mesh.NumNodes[2] - 1 })*//*909*/314);
			foreach (int nodeID in loadedNodes)
			{
				Node node = model.NodesDictionary[nodeID];
				model.Loads.Add(new Load() { Node = node, DOF = StructuralDof.TranslationZ, Amount = load });
			}

			return model;
		}

		public static IModel CreateMultiSubdomainModel()
		{
			Dictionary<int, int> elementsToSubdomains = GetSubdomainsOfElements();
			Model model = CreateSingleSubdomainModel();
			model.DecomposeIntoSubdomains(NumSubdomains[0] * NumSubdomains[1] * NumSubdomains[2], e => elementsToSubdomains[e]);
			return model;
		}

		public static ICornerDofSelection GetCornerDofs(IModel model) => UniformDdmModelBuilder3D.FindCornerDofs(model);
		

		public static NodalResults GetExpectedNodalValues(ActiveDofs allDofs)
		{
			int dofX = allDofs.GetIdOfDof(StructuralDof.TranslationX);
			int dofY = allDofs.GetIdOfDof(StructuralDof.TranslationY);
			int dofZ = allDofs.GetIdOfDof(StructuralDof.TranslationZ);

			var results = new Table<int, int, double>();
			#region long list of solution values per dof
			results[0, dofX] = 0; results[0, dofY] = 0; results[0, dofZ] = 0;
			results[1, dofX] = 50.922197735894756; results[1, dofY] = 17.9755448390772; results[1, dofZ] = -214.64651928346373;
			results[2, dofX] = 67.66259154588649; results[2, dofY] = 66.98508867950706; results[2, dofZ] = -24.789749258130566;
			results[3, dofX] = 47.87485771328109; results[3, dofY] = 116.28996057949249; results[3, dofZ] = 174.1355419208071;
			results[4, dofX] = 0; results[4, dofY] = 0; results[4, dofZ] = 0;
			results[5, dofX] = 43.9257432853672; results[5, dofY] = 8.396084637338213; results[5, dofZ] = -271.70186819630567;
			results[6, dofX] = 48.6177056802344; results[6, dofY] = 24.93499715188943; results[6, dofZ] = -50.884836062275845;
			results[7, dofX] = 53.97232337432083; results[7, dofY] = 68.66468339825879; results[7, dofZ] = 120.34542062694524;
			results[8, dofX] = 59.46447675017824; results[8, dofY] = 98.46378577221262; results[8, dofZ] = 292.4200336904981;
			results[9, dofX] = 47.546908199408364; results[9, dofY] = 95.24361708543412; results[9, dofZ] = 510.8077103197342;
			results[10, dofX] = 63.25666551024514; results[10, dofY] = 20.883384237658635; results[10, dofZ] = -142.67985307622726;
			results[11, dofX] = 59.09697248041097; results[11, dofY] = 36.39446177599841; results[11, dofZ] = 59.04846007667131;
			results[12, dofX] = 58.86547764291654; results[12, dofY] = 67.15292227957566; results[12, dofZ] = 264.95674456037335;
			results[13, dofX] = 61.1086426996612; results[13, dofY] = 95.633189365988; results[13, dofZ] = 470.65599127407495;
			results[14, dofX] = 64.00652592075276; results[14, dofY] = 109.81854240214388; results[14, dofZ] = 673.0239530019066;
			results[15, dofX] = 55.78320481447402; results[15, dofY] = 24.49789814481318; results[15, dofZ] = -20.45060213688484;
			results[16, dofX] = 55.86843434861002; results[16, dofY] = 41.2655139854666; results[16, dofZ] = 194.527523673131;
			results[17, dofX] = 56.11356204154401; results[17, dofY] = 67.25556381726635; results[17, dofZ] = 410.29925430922725;
			results[18, dofX] = 56.43801040768812; results[18, dofY] = 93.6124846278834; results[18, dofZ] = 625.9693574375175;
			results[19, dofX] = 55.993559737654664; results[19, dofY] = 111.16695224659368; results[19, dofZ] = 843.2201880529643;
			results[20, dofX] = 43.621844835787215; results[20, dofY] = 18.217014711088858; results[20, dofZ] = 103.29313033278129;
			results[21, dofX] = 48.94093761138207; results[21, dofY] = 37.01708945083352; results[21, dofZ] = 330.85234973387946;
			results[22, dofX] = 47.98677455699289; results[22, dofY] = 66.32871785196974; results[22, dofZ] = 552.9140156762392;
			results[23, dofX] = 46.39862716835806; results[23, dofY] = 92.78103569590203; results[23, dofZ] = 777.5475896657557;
			results[24, dofX] = 45.53402087819768; results[24, dofY] = 112.75424871524999; results[24, dofZ] = 1000.1288296513529;
			results[25, dofX] = 56.214537862158394; results[25, dofY] = 3.7452128677013827; results[25, dofZ] = 241.34551673921814;
			results[26, dofX] = 42.62116387907786; results[26, dofY] = 32.18686650265326; results[26, dofZ] = 441.58225556404875;
			results[27, dofX] = 37.04661225432036; results[27, dofY] = 65.7625661673989; results[27, dofZ] = 696.5158347114835;
			results[28, dofX] = 31.467285084843674; results[28, dofY] = 92.6782072502821; results[28, dofZ] = 925.1962106983615;
			results[29, dofX] = 30.017612164853837; results[29, dofY] = 113.49158184783475; results[29, dofZ] = 1152.5261240721895;
			results[30, dofX] = 0; results[30, dofY] = 0; results[30, dofZ] = 0;
			results[31, dofX] = 22.61072150582391; results[31, dofY] = 19.437273282814317; results[31, dofZ] = 604.400367164636;
			results[32, dofX] = 15.851054443334899; results[32, dofY] = 70.14538755065638; results[32, dofZ] = 833.6030901727183;
			results[33, dofX] = 11.157941840508473; results[33, dofY] = 92.46090112926737; results[33, dofZ] = 1072.5250245780794;
			results[34, dofX] = 10.42458733216261; results[34, dofY] = 114.15047975645629; results[34, dofZ] = 1299.945312272757;
			results[35, dofX] = -108.44049403578094; results[35, dofY] = -108.25752646962336; results[35, dofZ] = -317.0825148489985;
			results[36, dofX] = -86.51444373280957; results[36, dofY] = -94.24209706303351; results[36, dofZ] = -186.65458880437737;
			results[37, dofX] = -106.63357266409837; results[37, dofY] = -75.50490960955426; results[37, dofZ] = -24.239334637118194;
			results[38, dofX] = -92.86992635929467; results[38, dofY] = -65.41550329335548; results[38, dofZ] = 144.97642351155113;
			results[39, dofX] = -116.26649765648577; results[39, dofY] = -42.23411324687882; results[39, dofZ] = 283.31066463244116;
			results[40, dofX] = -130.55176441685282; results[40, dofY] = -82.701141639971; results[40, dofZ] = -241.17162398635986;
			results[41, dofX] = -126.55488132154827; results[41, dofY] = -92.02464775734637; results[41, dofZ] = -67.2234711447693;
			results[42, dofX] = -123.20763004226183; results[42, dofY] = -75.51415778680524; results[42, dofZ] = 119.15540593687683;
			results[43, dofX] = -125.56891148043965; results[43, dofY] = -61.693588425314715; results[43, dofZ] = 306.1579437667135;
			results[44, dofX] = -129.11159493929813; results[44, dofY] = -76.44967997549779; results[44, dofZ] = 474.4174601048281;
			results[45, dofX] = -145.8208623302301; results[45, dofY] = -105.43342233834505; results[45, dofZ] = -149.6485416276871;
			results[46, dofX] = -142.32826408643712; results[46, dofY] = -91.62415025850693; results[46, dofZ] = 59.24577809885651;
			results[47, dofX] = -142.18486963934603; results[47, dofY] = -75.34947746254969; results[47, dofZ] = 264.4124329022108;
			results[48, dofX] = -140.96946224071812; results[48, dofY] = -60.50169168540455; results[48, dofZ] = 468.55394972550255;
			results[49, dofX] = -145.46100569701954; results[49, dofY] = -48.091423928558285; results[49, dofZ] = 677.6318389738061;
			results[50, dofX] = -158.72140742615082; results[50, dofY] = -108.86185946465282; results[50, dofZ] = -20.630945897492815;
			results[51, dofX] = -158.04465641830473; results[51, dofY] = -93.33315506116294; results[51, dofZ] = 194.31724708758807;
			results[52, dofX] = -157.8489795152322; results[52, dofY] = -74.63335359276108; results[52, dofZ] = 410.1227007417851;
			results[53, dofX] = -158.4445650424871; results[53, dofY] = -56.06378814099574; results[53, dofZ] = 626.3755682560261;
			results[54, dofX] = -159.915529267852; results[54, dofY] = -39.35052327385978; results[54, dofZ] = 841.5777194288856;
			results[55, dofX] = -173.42436114908986; results[55, dofY] = -108.94844575831739; results[55, dofZ] = 110.2834410767148;
			results[56, dofX] = -175.98712982823136; results[56, dofY] = -91.28636853820977; results[56, dofZ] = 329.5397227631788;
			results[57, dofX] = -175.74615254607053; results[57, dofY] = -73.91074055981704; results[57, dofZ] = 554.2312229218331;
			results[58, dofX] = -176.406999704888; results[58, dofY] = -53.71401569412017; results[58, dofZ] = 777.8745772751224;
			results[59, dofX] = -177.40243262067364; results[59, dofY] = -34.93072275100292; results[59, dofZ] = 1000.1216735236978;
			results[60, dofX] = -198.13009286428587; results[60, dofY] = -89.6824497229456; results[60, dofZ] = 209.25468485291955;
			results[61, dofX] = -197.96499345677177; results[61, dofY] = -91.96010304921751; results[61, dofZ] = 456.96470477878137;
			results[62, dofX] = -198.5447278354371; results[62, dofY] = -72.9328367130458; results[62, dofZ] = 695.6452870396529;
			results[63, dofX] = -196.56724475826803; results[63, dofY] = -53.06628517766403; results[63, dofZ] = 926.5230483376528;
			results[64, dofX] = -196.96535166746344; results[64, dofY] = -32.57405676825995; results[64, dofZ] = 1152.4366593230006;
			results[65, dofX] = -215.69030286499813; results[65, dofY] = -115.94576241557621; results[65, dofZ] = 291.68577802721217;
			results[66, dofX] = -243.82812503645874; results[66, dofY] = -93.30309806339729; results[66, dofZ] = 571.362719955756;
			results[67, dofX] = -221.491908519286; results[67, dofY] = -76.2429634801263; results[67, dofZ] = 840.4991408594915;
			results[68, dofX] = -218.4675836452834; results[68, dofY] = -53.14682346436332; results[68, dofZ] = 1071.8126938472487;
			results[69, dofX] = -217.9139951379428; results[69, dofY] = -32.0915005998845; results[69, dofZ] = 1300.1713228044282;
			results[70, dofX] = -309.15497051497977; results[70, dofY] = -259.96716500801926; results[70, dofZ] = -409.45212822219685;
			results[71, dofX] = -299.56066746041733; results[71, dofY] = -237.76843094782984; results[71, dofZ] = -212.31147479304983;
			results[72, dofX] = -295.10380794026474; results[72, dofY] = -219.35243468109232; results[72, dofZ] = -25.61429410982252;
			results[73, dofX] = -299.36290445736887; results[73, dofY] = -199.7445700382937; results[73, dofZ] = 162.89704021302413;
			results[74, dofX] = -305.5687217330426; results[74, dofY] = -178.33780693368183; results[74, dofZ] = 359.783027560013;
			results[75, dofX] = -329.4988373081947; results[75, dofY] = -250.77222373782652; results[75, dofZ] = -272.6411386398248;
			results[76, dofX] = -322.05350613848105; results[76, dofY] = -232.04131902512918; results[76, dofZ] = -75.8856383732122;
			results[77, dofX] = -321.3185529180286; results[77, dofY] = -218.16479953086994; results[77, dofZ] = 118.41970862562935;
			results[78, dofX] = -321.67712956930654; results[78, dofY] = -204.0278477475847; results[78, dofZ] = 312.36843203232513;
			results[79, dofX] = -329.0649562325234; results[79, dofY] = -183.4284066871534; results[79, dofZ] = 509.2074870119579;
			results[80, dofX] = -351.68849955359025; results[80, dofY] = -248.46540026455406; results[80, dofZ] = -150.5664855486981;
			results[81, dofX] = -348.59993277163943; results[81, dofY] = -232.50664632790813; results[81, dofZ] = 56.799759473204546;
			results[82, dofX] = -347.3471394487668; results[82, dofY] = -217.4776591358973; results[82, dofZ] = 263.95942703776814;
			results[83, dofX] = -348.747744818727; results[83, dofY] = -202.27546510102604; results[83, dofZ] = 470.8834081781303;
			results[84, dofX] = -352.29506321006244; results[84, dofY] = -185.61108001043138; results[84, dofZ] = 677.6870796013512;
			results[85, dofX] = -373.01359333915906; results[85, dofY] = -251.82531377433185; results[85, dofZ] = -22.444991192355058;
			results[86, dofX] = -372.0757338038895; results[86, dofY] = -232.93831090705484; results[86, dofZ] = 193.51806779266246;
			results[87, dofX] = -372.0328791951983; results[87, dofY] = -216.44929687746836; results[87, dofZ] = 410.26230885679087;
			results[88, dofX] = -372.9984598039153; results[88, dofY] = -199.33683052671304; results[88, dofZ] = 626.8114639220503;
			results[89, dofX] = -375.67085018344005; results[89, dofY] = -179.21542658334903; results[89, dofZ] = 843.5470305798735;
			results[90, dofX] = -395.0100502226465; results[90, dofY] = -249.2124441272309; results[90, dofZ] = 107.26795190670644;
			results[91, dofX] = -395.92194514006263; results[91, dofY] = -232.0181863687371; results[91, dofZ] = 330.5442261378406;
			results[92, dofX] = -396.59026056461323; results[92, dofY] = -215.27835899164944; results[92, dofZ] = 555.2141441005374;
			results[93, dofX] = -397.2715756549937; results[93, dofY] = -197.00290538064343; results[93, dofZ] = 779.197164005205;
			results[94, dofX] = -398.8338550614058; results[94, dofY] = -175.95490160980455; results[94, dofZ] = 1000.9612954940737;
			results[95, dofX] = -417.4524090810113; results[95, dofY] = -250.90764079263036; results[95, dofZ] = 233.2428096641266;
			results[96, dofX] = -422.848336644583; results[96, dofY] = -231.05260649060673; results[96, dofZ] = 463.1796043553014;
			results[97, dofX] = -422.1915970959725; results[97, dofY] = -215.70204673997296; results[97, dofZ] = 698.3611623144892;
			results[98, dofX] = -421.612274784597; results[98, dofY] = -196.5252511755765; results[98, dofZ] = 928.1668715629339;
			results[99, dofX] = -422.12655226805634; results[99, dofY] = -175.09919206891064; results[99, dofZ] = 1153.946889645317;
			results[100, dofX] = -438.69033549684747; results[100, dofY] = -256.73170360099755; results[100, dofZ] = 369.64697063749037;
			results[101, dofX] = -444.65865822602615; results[101, dofY] = -237.04530453053002; results[101, dofZ] = 599.4968020142967;
			results[102, dofX] = -447.16962259804797; results[102, dofY] = -217.97106267810997; results[102, dofZ] = 840.1738513650944;
			results[103, dofX] = -444.3806244370263; results[103, dofY] = -197.97579575749887; results[103, dofZ] = 1074.6157810048583;
			results[104, dofX] = -444.89938318659676; results[104, dofY] = -175.74163369270153; results[104, dofZ] = 1301.1876459068596;
			results[105, dofX] = -516.7963608932072; results[105, dofY] = -412.9588083482648; results[105, dofZ] = -443.476443437731;
			results[106, dofX] = -510.21993551167543; results[106, dofY] = -387.9830868385917; results[106, dofZ] = -234.83167644567718;
			results[107, dofX] = -507.7119604371998; results[107, dofY] = -364.06891340584815; results[107, dofZ] = -29.727217832945062;
			results[108, dofX] = -508.04044683891345; results[108, dofY] = -339.7467577146955; results[108, dofZ] = 175.3507546226506;
			results[109, dofX] = -512.1477400503748; results[109, dofY] = -312.3869598224287; results[109, dofZ] = 383.5724136784895;
			results[110, dofX] = -540.7492052682069; results[110, dofY] = -406.4049534625002; results[110, dofZ] = -298.76178929465715;
			results[111, dofX] = -535.4950244953905; results[111, dofY] = -383.7066260828926; results[111, dofZ] = -88.63542704749298;
			results[112, dofX] = -533.1932155469775; results[112, dofY] = -362.88059792046556; results[112, dofZ] = 116.65110825808557;
			results[113, dofX] = -534.6823657254374; results[113, dofY] = -341.4425555219345; results[113, dofZ] = 322.0032735476348;
			results[114, dofX] = -538.9369221287235; results[114, dofY] = -317.0879381619898; results[114, dofZ] = 531.6949778632882;
			results[115, dofX] = -565.0447077662446; results[115, dofY] = -403.43188081246797; results[115, dofZ] = -161.82013890614988;
			results[116, dofX] = -561.9395498482967; results[116, dofY] = -381.3528406187113; results[116, dofZ] = 52.01685358263141;
			results[117, dofX] = -561.1949602316938; results[117, dofY] = -362.0007563681938; results[117, dofZ] = 263.64356127469;
			results[118, dofX] = -562.2913244092351; results[118, dofY] = -342.08658666231605; results[118, dofZ] = 475.1792542280515;
			results[119, dofX] = -565.9091269785949; results[119, dofY] = -318.19307471223254; results[119, dofZ] = 689.0325480165693;
			results[120, dofX] = -589.6916274768047; results[120, dofY] = -401.92997373476464; results[120, dofZ] = -26.314668552993844;
			results[121, dofX] = -588.6495266802177; results[121, dofY] = -380.4798316798842; results[121, dofZ] = 192.2680421675966;
			results[122, dofX] = -588.7539265032387; results[122, dofY] = -361.0495790381262; results[122, dofZ] = 411.0266136227567;
			results[123, dofX] = -590.1630253718257; results[123, dofY] = -340.929794549617; results[123, dofZ] = 629.6999075165576;
			results[124, dofX] = -592.9383085520374; results[124, dofY] = -317.480097809357; results[124, dofZ] = 848.1951219728622;
			results[125, dofX] = -613.8191244892316; results[125, dofY] = -401.68284882003786; results[125, dofZ] = 110.08227981873458;
			results[126, dofX] = -614.7620266669986; results[126, dofY] = -380.01235090162487; results[126, dofZ] = 332.7249543362268;
			results[127, dofX] = -615.6373967763052; results[127, dofY] = -360.50315104966086; results[127, dofZ] = 558.0435230164044;
			results[128, dofX] = -617.164403189857; results[128, dofY] = -339.948971091149; results[128, dofZ] = 782.7403025364447;
			results[129, dofX] = -619.5717167167716; results[129, dofY] = -316.31588730216544; results[129, dofZ] = 1005.4281354500682;
			results[130, dofX] = -636.7231849942982; results[130, dofY] = -402.36268756732034; results[130, dofZ] = 248.65055040061083;
			results[131, dofX] = -639.5623442031161; results[131, dofY] = -381.44363098094254; results[131, dofZ] = 473.79504444424884;
			results[132, dofX] = -641.6875827079841; results[132, dofY] = -361.3230252651464; results[132, dofZ] = 704.0398412753408;
			results[133, dofX] = -643.1668295662021; results[133, dofY] = -340.54099917604185; results[133, dofZ] = 933.4607977813191;
			results[134, dofX] = -645.3482641442434; results[134, dofY] = -317.1095083242753; results[134, dofZ] = 1158.999593546596;
			results[135, dofX] = -657.5661664807868; results[135, dofY] = -406.19516200672916; results[135, dofZ] = 393.0115712909252;
			results[136, dofX] = -662.2237197139825; results[136, dofY] = -385.0615650024591; results[136, dofZ] = 619.2642264711818;
			results[137, dofX] = -664.8316805492183; results[137, dofY] = -364.30586326674944; results[137, dofZ] = 850.4605757543302;
			results[138, dofX] = -667.2455431108596; results[138, dofY] = -342.8152028556309; results[138, dofZ] = 1080.872118878769;
			results[139, dofX] = -669.6466407355708; results[139, dofY] = -319.4944574350215; results[139, dofZ] = 1307.4736017741693;
			results[140, dofX] = -731.2372008159687; results[140, dofY] = -566.285550149535; results[140, dofZ] = -464.7088975645027;
			results[141, dofX] = -727.1962230559633; results[141, dofY] = -539.1075460947034; results[141, dofZ] = -248.44054799456563;
			results[142, dofX] = -725.0572418980151; results[142, dofY] = -512.0518724929176; results[142, dofZ] = -34.471072574311194;
			results[143, dofX] = -724.8519294162007; results[143, dofY] = -484.39092173047004; results[143, dofZ] = 179.39168043964037;
			results[144, dofX] = -726.3237196245608; results[144, dofY] = -455.2433569111271; results[144, dofZ] = 395.6493380961883;
			results[145, dofX] = -757.792739023881; results[145, dofY] = -561.8666638303907; results[145, dofZ] = -315.89324892476185;
			results[146, dofX] = -754.2291398120551; results[146, dofY] = -536.0192928649087; results[146, dofZ] = -99.22997815518015;
			results[147, dofX] = -752.6667244827015; results[147, dofY] = -510.827620107346; results[147, dofZ] = 114.47933747404775;
			results[148, dofX] = -753.1977090761919; results[148, dofY] = -485.08592386470383; results[148, dofZ] = 328.1223833724118;
			results[149, dofX] = -755.774060274764; results[149, dofY] = -457.33096337927157; results[149, dofZ] = 544.7294733727408;
			results[150, dofX] = -784.0176218479776; results[150, dofY] = -558.7211228001521; results[150, dofZ] = -172.84732335520795;
			results[151, dofX] = -781.6710039684144; results[151, dofY] = -533.721239971761; results[151, dofZ] = 46.29708233548891;
			results[152, dofX] = -780.9419278769354; results[152, dofY] = -509.84002571443824; results[152, dofZ] = 263.1586844815226;
			results[153, dofX] = -781.9650695796835; results[153, dofY] = -485.3309209847675; results[153, dofZ] = 479.9846863231244;
			results[154, dofX] = -784.6385967961919; results[154, dofY] = -458.4582520862555; results[154, dofZ] = 698.9278555729833;
			results[155, dofX] = -810.0353956214636; results[155, dofY] = -556.787139240982; results[155, dofZ] = -31.420966360983755;
			results[156, dofX] = -809.0039162048278; results[156, dofY] = -532.2816575520726; results[156, dofZ] = 190.60734802153556;
			results[157, dofX] = -809.147953551496; results[157, dofY] = -508.90928929638926; results[157, dofZ] = 412.27487550227215;
			results[158, dofX] = -810.5184690443127; results[158, dofY] = -484.7905606108231; results[158, dofZ] = 633.9211181036097;
			results[159, dofX] = -813.2926702910822; results[159, dofY] = -458.28060270271015; results[159, dofZ] = 855.7722951200694;
			results[160, dofX] = -835.1145317524131; results[160, dofY] = -555.6467574210305; results[160, dofZ] = 110.61881910637555;
			results[161, dofX] = -835.3438444670319; results[161, dofY] = -531.8946525877922; results[161, dofZ] = 335.4099559931488;
			results[162, dofX] = -836.3108590846068; results[162, dofY] = -508.5551297130784; results[162, dofZ] = 562.0059093443333;
			results[163, dofX] = -838.2280829249183; results[163, dofY] = -484.4849509801514; results[163, dofZ] = 788.6974924953195;
			results[164, dofX] = -841.3522054433568; results[164, dofY] = -458.444784106826; results[164, dofZ] = 1013.3654079685898;
			results[165, dofX] = -858.7078351792071; results[165, dofY] = -555.8182553333753; results[165, dofZ] = 254.96364881792147;
			results[166, dofX] = -860.1158072306202; results[166, dofY] = -532.8659646690209; results[166, dofZ] = 481.9183225372855;
			results[167, dofX] = -861.9813354473388; results[167, dofY] = -509.7924036029123; results[167, dofZ] = 712.1401797193068;
			results[168, dofX] = -864.6496889979326; results[168, dofY] = -485.90918945509037; results[168, dofZ] = 942.4958519311733;
			results[169, dofX] = -868.4139774604832; results[169, dofY] = -460.50510383649123; results[169, dofZ] = 1169.4948475080091;
			results[170, dofX] = -880.891506624111; results[170, dofY] = -557.1815565498089; results[170, dofZ] = 404.05186626249724;
			results[171, dofX] = -883.1973379563707; results[171, dofY] = -535.3808827423663; results[171, dofZ] = 631.4444196828331;
			results[172, dofX] = -885.9898554950956; results[172, dofY] = -512.7767718280663; results[172, dofZ] = 862.007406117013;
			results[173, dofX] = -889.4887170701693; results[173, dofY] = -489.39364305316576; results[173, dofZ] = 1092.7429686910868;
			results[174, dofX] = -894.1846709686329; results[174, dofY] = -464.7127499738378; results[174, dofZ] = 1320.2397927062646;
			results[175, dofX] = -950.9239705552658; results[175, dofY] = -720.5151958888416; results[175, dofZ] = -478.1256352694338;
			results[176, dofX] = -948.3383523347445; results[176, dofY] = -691.6814221114755; results[176, dofZ] = -257.5108403038448;
			results[177, dofX] = -946.873260693239; results[177, dofY] = -662.8350316979663; results[177, dofZ] = -38.61101557243818;
			results[178, dofX] = -946.391585009902; results[178, dofY] = -633.531992062616; results[178, dofZ] = 180.13539651467573;
			results[179, dofX] = -946.9426321500307; results[179, dofY] = -603.1854178188187; results[179, dofZ] = 400.95079436854667;
			results[180, dofX] = -979.2041956571754; results[180, dofY] = -717.5978117695211; results[180, dofZ] = -327.50227713492404;
			results[181, dofX] = -976.9389383240036; results[181, dofY] = -689.633017868615; results[181, dofZ] = -106.47253875116785;
			results[182, dofX] = -975.8258052110441; results[182, dofY] = -661.8497899325355; results[182, dofZ] = 112.32787910464153;
			results[183, dofX] = -975.990984126009; results[183, dofY] = -633.5939724623881; results[183, dofZ] = 331.07919414978534;
			results[184, dofX] = -977.3406543267977; results[184, dofY] = -604.1488420129995; results[184, dofZ] = 552.18345097998;
			results[185, dofX] = -1007.1354232468555; results[185, dofY] = -715.2436283452736; results[185, dofZ] = -181.39136347988216;
			results[186, dofX] = -1005.4783907277806; results[186, dofY] = -687.8986851956968; results[186, dofZ] = 41.56771529726331;
			results[187, dofX] = -1004.9174314286734; results[187, dofY] = -660.9356163135197; results[187, dofZ] = 262.45632739794195;
			results[188, dofX] = -1005.5294134322752; results[188, dofY] = -633.382708719057; results[188, dofZ] = 483.25315985358736;
			results[189, dofX] = -1007.3696577343331; results[189, dofY] = -604.4344344117316; results[189, dofZ] = 705.8065192965113;
			results[190, dofX] = -1034.5488145523593; results[190, dofY] = -713.3029076715789; results[190, dofZ] = -36.724876352270925;
			results[191, dofX] = -1033.6002851981832; results[191, dofY] = -686.6060502805265; results[191, dofZ] = 188.76170140922196;
			results[192, dofX] = -1033.5641994569812; results[192, dofY] = -659.9477435086983; results[192, dofZ] = 413.2340959134017;
			results[193, dofX] = -1034.6523975463742; results[193, dofY] = -632.4090899984719; results[193, dofZ] = 637.7575656956235;
			results[194, dofX] = -1036.9969985595158; results[194, dofY] = -603.528317594143; results[194, dofZ] = 862.8575212582948;
			results[195, dofX] = -1060.7600099056174; results[195, dofY] = -711.8470263927653; results[195, dofZ] = 108.53305090569891;
			results[196, dofX] = -1060.3827440098669; results[196, dofY] = -685.9750537115449; results[196, dofZ] = 336.7659525826029;
			results[197, dofX] = -1060.7308560895679; results[197, dofY] = -659.397272351548; results[197, dofZ] = 565.6617349031907;
			results[198, dofX] = -1062.1655017870762; results[198, dofY] = -631.4762421876574; results[198, dofZ] = 795.4682871175243;
			results[199, dofX] = -1065.5979600225237; results[199, dofY] = -603.0975138367908; results[199, dofZ] = 1023.8839543879337;
			results[200, dofX] = -1085.4045376163867; results[200, dofY] = -711.1335301266591; results[200, dofZ] = 256.0073148919006;
			results[201, dofX] = -1085.374721768517; results[201, dofY] = -686.4263488349135; results[201, dofZ] = 486.69172347872126;
			results[202, dofX] = -1085.8221383497123; results[202, dofY] = -660.2198692069242; results[202, dofZ] = 719.9574022964337;
			results[203, dofX] = -1088.1642241028935; results[203, dofY] = -632.958491810904; results[203, dofZ] = 955.0466000375958;
			results[204, dofX] = -1093.4966774978204; results[204, dofY] = -605.4707155224448; results[204, dofZ] = 1186.0080174540294;
			results[205, dofX] = -1108.8210884315158; results[205, dofY] = -711.323660755969; results[205, dofZ] = 407.7854069762592;
			results[206, dofX] = -1109.2289064977415; results[206, dofY] = -688.0197792573376; results[206, dofZ] = 639.3509443357237;
			results[207, dofX] = -1110.4506435370142; results[207, dofY] = -663.2809743849647; results[207, dofZ] = 873.7933276674635;
			results[208, dofX] = -1113.7713118990475; results[208, dofY] = -637.9503675793363; results[208, dofZ] = 1109.706868138546;
			results[209, dofX] = -1120.4704513116021; results[209, dofY] = -611.8773581488617; results[209, dofZ] = 1341.9528302947983;
			results[210, dofX] = -1173.7254992314422; results[210, dofY] = -875.2971332650636; results[210, dofZ] = -485.92209920444776;
			results[211, dofX] = -1172.3893767577933; results[211, dofY] = -845.3913021410394; results[211, dofZ] = -262.9301906768495;
			results[212, dofX] = -1171.5149045245712; results[212, dofY] = -815.7122073525535; results[212, dofZ] = -41.46992071924802;
			results[213, dofX] = -1171.0642593514551; results[213, dofY] = -785.7909499805062; results[213, dofZ] = 179.81392693092312;
			results[214, dofX] = -1171.116086332369; results[214, dofY] = -755.190641443248; results[214, dofZ] = 403.01431728471334;
			results[215, dofX] = -1203.523135984145; results[215, dofY] = -873.8499776998053; results[215, dofZ] = -334.5854298187874;
			results[216, dofX] = -1202.1828286242944; results[216, dofY] = -844.3128511250956; results[216, dofZ] = -111.05930079535206;
			results[217, dofX] = -1201.4221356445448; results[217, dofY] = -815.2170123204448; results[217, dofZ] = 110.65064579998287;
			results[218, dofX] = -1201.3163889055534; results[218, dofY] = -785.8223507647646; results[218, dofZ] = 332.29604560222657;
			results[219, dofX] = -1201.86901433943; results[219, dofY] = -755.5018366552231; results[219, dofZ] = 555.7900147614678;
			results[220, dofX] = -1233.3712920433081; results[220, dofY] = -872.6419775886748; results[220, dofZ] = -187.1601944467686;
			results[221, dofX] = -1232.3072111738368; results[221, dofY] = -843.4110308513818; results[221, dofZ] = 38.27675918735277;
			results[222, dofX] = -1231.7699931221046; results[222, dofY] = -814.8358138092048; results[222, dofZ] = 261.66675011907967;
			results[223, dofX] = -1231.916939038852; results[223, dofY] = -785.8005061929861; results[223, dofZ] = 484.8665344742888;
			results[224, dofX] = -1232.7677008031312; results[224, dofY] = -755.413800940861; results[224, dofZ] = 709.6794652194128;
			results[225, dofX] = -1263.0173199469552; results[225, dofY] = -871.3793043181485; results[225, dofZ] = -41.152650518738604;
			results[226, dofX] = -1262.2908067391697; results[226, dofY] = -842.7032470356567; results[226, dofZ] = 187.02509621489898;
			results[227, dofX] = -1262.0106276101822; results[227, dofY] = -814.386967198193; results[227, dofZ] = 413.45323689411487;
			results[228, dofX] = -1262.3890912008862; results[228, dofY] = -784.9465690337673; results[228, dofZ] = 639.8567908849849;
			results[229, dofX] = -1263.9628385783167; results[229, dofY] = -754.3818705697016; results[229, dofZ] = 866.806800482139;
			results[230, dofX] = -1291.4629907476142; results[230, dofY] = -869.8719460783839; results[230, dofZ] = 105.38749075410357;
			results[231, dofX] = -1290.913611335645; results[231, dofY] = -842.2305427697892; results[231, dofZ] = 336.7343436167181;
			results[232, dofX] = -1290.5468922996617; results[232, dofY] = -813.9172174042318; results[232, dofZ] = 567.7921343707743;
			results[233, dofX] = -1291.3968977781062; results[233, dofY] = -783.0964466520851; results[233, dofZ] = 800.1626404501585;
			results[234, dofX] = -1294.1380523244854; results[234, dofY] = -750.2184405674698; results[234, dofZ] = 1033.5845667215383;
			results[235, dofX] = -1318.0442019292966; results[235, dofY] = -868.6673490027099; results[235, dofZ] = 254.11152280356654;
			results[236, dofX] = -1317.1609941865254; results[236, dofY] = -842.2599138495204; results[236, dofZ] = 488.4914286063462;
			results[237, dofX] = -1315.5966077208648; results[237, dofY] = -814.66443806261; results[237, dofZ] = 724.9340075637762;
			results[238, dofX] = -1314.9430786301252; results[238, dofY] = -782.2624995143696; results[238, dofZ] = 968.4204251428021;
			results[239, dofX] = -1321.1495243353124; results[239, dofY] = -751.466537077772; results[239, dofZ] = 1212.036180271913;
			results[240, dofX] = -1342.9812864867222; results[240, dofY] = -867.9994707645498; results[240, dofZ] = 407.5966689113463;
			results[241, dofX] = -1342.2273655513197; results[241, dofY] = -843.314344440923; results[241, dofZ] = 642.6427788911732;
			results[242, dofX] = -1338.453518393455; results[242, dofY] = -817.2892306802516; results[242, dofZ] = 883.6819901840354;
			results[243, dofX] = -1339.831202815678; results[243, dofY] = -788.4347338628578; results[243, dofZ] = 1135.8188767603;
			results[244, dofX] = -1351.5583306148103; results[244, dofY] = -763.1652922876309; results[244, dofZ] = 1379.2713616998744;
			results[245, dofX] = -1398.046821030171; results[245, dofY] = -1029.4941951238648; results[245, dofZ] = -489.0757465303704;
			results[246, dofX] = -1397.6689173796942; results[246, dofY] = -998.968579960984; results[246, dofZ] = -265.26330068036947;
			results[247, dofX] = -1397.3401436994575; results[247, dofY] = -969.3264090691325; results[247, dofZ] = -42.77524981549544;
			results[248, dofX] = -1396.8402906645788; results[248, dofY] = -939.6381206044736; results[248, dofZ] = 179.59218139824077;
			results[249, dofX] = -1396.5776233166794; results[249, dofY] = -909.1985121057845; results[249, dofZ] = 403.49749792668405;
			results[250, dofX] = -1429.0305372293128; results[250, dofY] = -1029.3642599727364; results[250, dofZ] = -337.8775746296373;
			results[251, dofX] = -1428.3659287498476; results[251, dofY] = -998.7272829826442; results[251, dofZ] = -113.38729773316587;
			results[252, dofX] = -1427.8319861592; results[252, dofY] = -969.4755461972404; results[252, dofZ] = 109.74984526982254;
			results[253, dofX] = -1427.446568803795; results[253, dofY] = -940.0858776710155; results[253, dofZ] = 332.76671534701677;
			results[254, dofX] = -1427.3381702428007; results[254, dofY] = -909.4627152219647; results[254, dofZ] = 557.0845025023897;
			results[255, dofX] = -1461.1784881224553; results[255, dofY] = -1029.6371357707333; results[255, dofZ] = -189.9195942644658;
			results[256, dofX] = -1460.5740056873833; results[256, dofY] = -998.8290085406178; results[256, dofZ] = 36.57006980009705;
			results[257, dofX] = -1460.0260219479592; results[257, dofY] = -970.31393015184; results[257, dofZ] = 261.27950291626377;
			results[258, dofX] = -1459.3832529165513; results[258, dofY] = -941.3357808676094; results[258, dofZ] = 485.7531039731349;
			results[259, dofX] = -1459.1345244105948; results[259, dofY] = -910.1394876039524; results[259, dofZ] = 711.0526484747992;
			results[260, dofX] = -1494.0904519193439; results[260, dofY] = -1029.4993184889104; results[260, dofZ] = -43.64781132709869;
			results[261, dofX] = -1493.8460751964928; results[261, dofY] = -999.277791854101; results[261, dofZ] = 185.97424616600355;
			results[262, dofX] = -1493.3722768025814; results[262, dofY] = -971.8356445643496; results[262, dofZ] = 413.5093874988979;
			results[263, dofX] = -1492.7882951130218; results[263, dofY] = -943.5566825446875; results[263, dofZ] = 640.5514377210246;
			results[264, dofX] = -1492.2620117426652; results[264, dofY] = -911.0591727112604; results[264, dofZ] = 869.0439483924207;
			results[265, dofX] = -1526.2469121815925; results[265, dofY] = -1028.3778540713956; results[265, dofZ] = 103.02929260984422;
			results[266, dofX] = -1526.44612310151; results[266, dofY] = -999.4908593139331; results[266, dofZ] = 336.2752069925747;
			results[267, dofX] = -1527.0895153415336; results[267, dofY] = -973.7616421589253; results[267, dofZ] = 568.8032474324733;
			results[268, dofX] = -1526.8003034796536; results[268, dofY] = -944.8077389173399; results[268, dofZ] = 802.1404267645231;
			results[269, dofX] = -1528.501210301288; results[269, dofY] = -911.4517704047898; results[269, dofZ] = 1033.2384973888113;
			results[270, dofX] = -1555.7125608149342; results[270, dofY] = -1026.5386941328734; results[270, dofZ] = 252.28359235313823;
			results[271, dofX] = -1556.3011337327384; results[271, dofY] = -999.4981086268659; results[271, dofZ] = 488.4216535644294;
			results[272, dofX] = -1556.1460278643028; results[272, dofY] = -974.3771511512546; results[272, dofZ] = 726.8039579097897;
			results[273, dofX] = -1557.901462566557; results[273, dofY] = -947.4695211881374; results[273, dofZ] = 977.2651126178608;
			results[274, dofX] = -1563.2739298853635; results[274, dofY] = -892.186062185834; results[274, dofZ] = 1240.3638397942948;
			results[275, dofX] = -1582.5632053268089; results[275, dofY] = -1025.4703648876532; results[275, dofZ] = 405.46891627378574;
			results[276, dofX] = -1582.0290476606403; results[276, dofY] = -999.2976311444345; results[276, dofZ] = 643.7818660310684;
			results[277, dofX] = -1580.8027183706101; results[277, dofY] = -976.6455423168995; results[277, dofZ] = 883.0826100657547;
			results[278, dofX] = -1560.3722006390672; results[278, dofY] = -953.334419053854; results[278, dofZ] = 1164.310712482019;
			results[279, dofX] = -1578.8693840298483; results[279, dofY] = -911.1389309818509; results[279, dofZ] = 1483.2270800764193;
			results[280, dofX] = -1622.319984766827; results[280, dofY] = -1181.7291175738235; results[280, dofZ] = -489.6663625441447;
			results[281, dofX] = -1622.2156959485083; results[281, dofY] = -1151.2519349632814; results[281, dofZ] = -265.8301406468163;
			results[282, dofX] = -1621.9398336555837; results[282, dofY] = -1122.4016308975104; results[282, dofZ] = -43.08410233621198;
			results[283, dofX] = -1621.5454130038434; results[283, dofY] = -1093.530966998705; results[283, dofZ] = 179.6020774007056;
			results[284, dofX] = -1621.3161969671442; results[284, dofY] = -1063.3927863311947; results[284, dofZ] = 403.54568298412363;
			results[285, dofX] = -1654.0647533857898; results[285, dofY] = -1181.740680016463; results[285, dofZ] = -338.71858200564236;
			results[286, dofX] = -1653.6140670885459; results[286, dofY] = -1151.0812657996435; results[286, dofZ] = -114.15342136661921;
			results[287, dofX] = -1652.9704214847795; results[287, dofY] = -1122.7993112711151; results[287, dofZ] = 109.57732329502946;
			results[288, dofX] = -1652.4909009983971; results[288, dofY] = -1094.296387400554; results[288, dofZ] = 333.179464914543;
			results[289, dofX] = -1652.1646182018465; results[289, dofY] = -1063.8905552041033; results[289, dofZ] = 557.4525202088945;
			results[290, dofX] = -1688.5667434263376; results[290, dofY] = -1182.0372167681446; results[290, dofZ] = -190.53779849505418;
			results[291, dofX] = -1688.0446666107662; results[291, dofY] = -1151.3067976747343; results[291, dofZ] = 36.16799403124678;
			results[292, dofX] = -1687.1398269022905; results[292, dofY] = -1124.5804207266583; results[292, dofZ] = 261.70037838428914;
			results[293, dofX] = -1685.933053786865; results[293, dofY] = -1097.1938972718588; results[293, dofZ] = 486.7549346704419;
			results[294, dofX] = -1684.6435171676378; results[294, dofY] = -1066.3027038485436; results[294, dofZ] = 712.2072708638335;
			results[295, dofX] = -1725.3137236189839; results[295, dofY] = -1182.4037412332796; results[295, dofZ] = -44.27972281534676;
			results[296, dofX] = -1725.095586999598; results[296, dofY] = -1152.7135005489868; results[296, dofZ] = 186.05364745377602;
			results[297, dofX] = -1724.2721692759915; results[297, dofY] = -1129.0548771783456; results[297, dofZ] = 414.5811931015931;
			results[298, dofX] = -1722.4369283020676; results[298, dofY] = -1104.209502663555; results[298, dofZ] = 642.9417703538221;
			results[299, dofX] = -1719.8833629806365; results[299, dofY] = -1072.417574967617; results[299, dofZ] = 869.6069958822934;
			results[300, dofX] = -1761.7952051670775; results[300, dofY] = -1182.2243608557085; results[300, dofZ] = 102.3561383312616;
			results[301, dofX] = -1762.8962281622148; results[301, dofY] = -1154.3351738606527; results[301, dofZ] = 336.7665003546309;
			results[302, dofX] = -1766.060963738986; results[302, dofY] = -1135.7043013931873; results[302, dofZ] = 571.4213476519601;
			results[303, dofX] = -1764.8098108426495; results[303, dofY] = -1119.3808769053505; results[303, dofZ] = 803.8511041470321;
			results[304, dofX] = -1758.6459434062556; results[304, dofY] = -1086.501617122351; results[304, dofZ] = 1042.4210659947937;
			results[305, dofX] = -1794.6492467726655; results[305, dofY] = -1181.3058562846124; results[305, dofZ] = 251.61103553914467;
			results[306, dofX] = -1797.1201398179242; results[306, dofY] = -1154.8699196150744; results[306, dofZ] = 489.8156601385115;
			results[307, dofX] = -1808.3973729189117; results[307, dofY] = -1136.711318874701; results[307, dofZ] = 728.1207423221688;
			results[308, dofX] = -1829.0270839707855; results[308, dofY] = -1141.9273500018705; results[308, dofZ] = 996.7704643908087;
			results[309, dofX] = -1826.1437006810518; results[309, dofY] = -1121.4555300572365; results[309, dofZ] = 1213.6702153840877;
			results[310, dofX] = -1822.8053790886195; results[310, dofY] = -1180.3800001950283; results[310, dofZ] = 404.9493180916093;
			results[311, dofX] = -1824.4732734655358; results[311, dofY] = -1154.0991539769466; results[311, dofZ] = 642.9847435410618;
			results[312, dofX] = -1834.2626968094442; results[312, dofY] = -1131.8603126604207; results[312, dofZ] = 891.6340067631387;
			results[313, dofX] = -1867.0999875569455; results[313, dofY] = -1139.7527343766856; results[313, dofZ] = 1137.5972602750649;
			results[314, dofX] = -1950.1928617927392; results[314, dofY] = -1205.0860441756715; results[314, dofZ] = 1833.4303677525677;
			#endregion

			return new NodalResults(results);
		}

		public static Dictionary<int, int> GetSubdomainsOfElements()
		{
			var elementsToSubdomains = new Dictionary<int, int>();
			#region long list of element -> subdomain associations
			elementsToSubdomains[0] = 0;
			elementsToSubdomains[1] = 0;
			elementsToSubdomains[2] = 1;
			elementsToSubdomains[3] = 1;
			elementsToSubdomains[4] = 0;
			elementsToSubdomains[5] = 0;
			elementsToSubdomains[6] = 1;
			elementsToSubdomains[7] = 1;
			elementsToSubdomains[8] = 2;
			elementsToSubdomains[9] = 2;
			elementsToSubdomains[10] = 3;
			elementsToSubdomains[11] = 3;
			elementsToSubdomains[12] = 2;
			elementsToSubdomains[13] = 2;
			elementsToSubdomains[14] = 3;
			elementsToSubdomains[15] = 3;
			elementsToSubdomains[16] = 4;
			elementsToSubdomains[17] = 4;
			elementsToSubdomains[18] = 5;
			elementsToSubdomains[19] = 5;
			elementsToSubdomains[20] = 4;
			elementsToSubdomains[21] = 4;
			elementsToSubdomains[22] = 5;
			elementsToSubdomains[23] = 5;
			elementsToSubdomains[24] = 0;
			elementsToSubdomains[25] = 0;
			elementsToSubdomains[26] = 1;
			elementsToSubdomains[27] = 1;
			elementsToSubdomains[28] = 0;
			elementsToSubdomains[29] = 0;
			elementsToSubdomains[30] = 1;
			elementsToSubdomains[31] = 1;
			elementsToSubdomains[32] = 2;
			elementsToSubdomains[33] = 2;
			elementsToSubdomains[34] = 3;
			elementsToSubdomains[35] = 3;
			elementsToSubdomains[36] = 2;
			elementsToSubdomains[37] = 2;
			elementsToSubdomains[38] = 3;
			elementsToSubdomains[39] = 3;
			elementsToSubdomains[40] = 4;
			elementsToSubdomains[41] = 4;
			elementsToSubdomains[42] = 5;
			elementsToSubdomains[43] = 5;
			elementsToSubdomains[44] = 4;
			elementsToSubdomains[45] = 4;
			elementsToSubdomains[46] = 5;
			elementsToSubdomains[47] = 5;
			elementsToSubdomains[48] = 6;
			elementsToSubdomains[49] = 6;
			elementsToSubdomains[50] = 7;
			elementsToSubdomains[51] = 7;
			elementsToSubdomains[52] = 6;
			elementsToSubdomains[53] = 6;
			elementsToSubdomains[54] = 7;
			elementsToSubdomains[55] = 7;
			elementsToSubdomains[56] = 8;
			elementsToSubdomains[57] = 8;
			elementsToSubdomains[58] = 9;
			elementsToSubdomains[59] = 9;
			elementsToSubdomains[60] = 8;
			elementsToSubdomains[61] = 8;
			elementsToSubdomains[62] = 9;
			elementsToSubdomains[63] = 9;
			elementsToSubdomains[64] = 10;
			elementsToSubdomains[65] = 10;
			elementsToSubdomains[66] = 11;
			elementsToSubdomains[67] = 11;
			elementsToSubdomains[68] = 10;
			elementsToSubdomains[69] = 10;
			elementsToSubdomains[70] = 11;
			elementsToSubdomains[71] = 11;
			elementsToSubdomains[72] = 6;
			elementsToSubdomains[73] = 6;
			elementsToSubdomains[74] = 7;
			elementsToSubdomains[75] = 7;
			elementsToSubdomains[76] = 6;
			elementsToSubdomains[77] = 6;
			elementsToSubdomains[78] = 7;
			elementsToSubdomains[79] = 7;
			elementsToSubdomains[80] = 8;
			elementsToSubdomains[81] = 8;
			elementsToSubdomains[82] = 9;
			elementsToSubdomains[83] = 9;
			elementsToSubdomains[84] = 8;
			elementsToSubdomains[85] = 8;
			elementsToSubdomains[86] = 9;
			elementsToSubdomains[87] = 9;
			elementsToSubdomains[88] = 10;
			elementsToSubdomains[89] = 10;
			elementsToSubdomains[90] = 11;
			elementsToSubdomains[91] = 11;
			elementsToSubdomains[92] = 10;
			elementsToSubdomains[93] = 10;
			elementsToSubdomains[94] = 11;
			elementsToSubdomains[95] = 11;
			elementsToSubdomains[96] = 12;
			elementsToSubdomains[97] = 12;
			elementsToSubdomains[98] = 13;
			elementsToSubdomains[99] = 13;
			elementsToSubdomains[100] = 12;
			elementsToSubdomains[101] = 12;
			elementsToSubdomains[102] = 13;
			elementsToSubdomains[103] = 13;
			elementsToSubdomains[104] = 14;
			elementsToSubdomains[105] = 14;
			elementsToSubdomains[106] = 15;
			elementsToSubdomains[107] = 15;
			elementsToSubdomains[108] = 14;
			elementsToSubdomains[109] = 14;
			elementsToSubdomains[110] = 15;
			elementsToSubdomains[111] = 15;
			elementsToSubdomains[112] = 16;
			elementsToSubdomains[113] = 16;
			elementsToSubdomains[114] = 17;
			elementsToSubdomains[115] = 17;
			elementsToSubdomains[116] = 16;
			elementsToSubdomains[117] = 16;
			elementsToSubdomains[118] = 17;
			elementsToSubdomains[119] = 17;
			elementsToSubdomains[120] = 12;
			elementsToSubdomains[121] = 12;
			elementsToSubdomains[122] = 13;
			elementsToSubdomains[123] = 13;
			elementsToSubdomains[124] = 12;
			elementsToSubdomains[125] = 12;
			elementsToSubdomains[126] = 13;
			elementsToSubdomains[127] = 13;
			elementsToSubdomains[128] = 14;
			elementsToSubdomains[129] = 14;
			elementsToSubdomains[130] = 15;
			elementsToSubdomains[131] = 15;
			elementsToSubdomains[132] = 14;
			elementsToSubdomains[133] = 14;
			elementsToSubdomains[134] = 15;
			elementsToSubdomains[135] = 15;
			elementsToSubdomains[136] = 16;
			elementsToSubdomains[137] = 16;
			elementsToSubdomains[138] = 17;
			elementsToSubdomains[139] = 17;
			elementsToSubdomains[140] = 16;
			elementsToSubdomains[141] = 16;
			elementsToSubdomains[142] = 17;
			elementsToSubdomains[143] = 17;
			elementsToSubdomains[144] = 18;
			elementsToSubdomains[145] = 18;
			elementsToSubdomains[146] = 19;
			elementsToSubdomains[147] = 19;
			elementsToSubdomains[148] = 18;
			elementsToSubdomains[149] = 18;
			elementsToSubdomains[150] = 19;
			elementsToSubdomains[151] = 19;
			elementsToSubdomains[152] = 20;
			elementsToSubdomains[153] = 20;
			elementsToSubdomains[154] = 21;
			elementsToSubdomains[155] = 21;
			elementsToSubdomains[156] = 20;
			elementsToSubdomains[157] = 20;
			elementsToSubdomains[158] = 21;
			elementsToSubdomains[159] = 21;
			elementsToSubdomains[160] = 22;
			elementsToSubdomains[161] = 22;
			elementsToSubdomains[162] = 23;
			elementsToSubdomains[163] = 23;
			elementsToSubdomains[164] = 22;
			elementsToSubdomains[165] = 22;
			elementsToSubdomains[166] = 23;
			elementsToSubdomains[167] = 23;
			elementsToSubdomains[168] = 18;
			elementsToSubdomains[169] = 18;
			elementsToSubdomains[170] = 19;
			elementsToSubdomains[171] = 19;
			elementsToSubdomains[172] = 18;
			elementsToSubdomains[173] = 18;
			elementsToSubdomains[174] = 19;
			elementsToSubdomains[175] = 19;
			elementsToSubdomains[176] = 20;
			elementsToSubdomains[177] = 20;
			elementsToSubdomains[178] = 21;
			elementsToSubdomains[179] = 21;
			elementsToSubdomains[180] = 20;
			elementsToSubdomains[181] = 20;
			elementsToSubdomains[182] = 21;
			elementsToSubdomains[183] = 21;
			elementsToSubdomains[184] = 22;
			elementsToSubdomains[185] = 22;
			elementsToSubdomains[186] = 23;
			elementsToSubdomains[187] = 23;
			elementsToSubdomains[188] = 22;
			elementsToSubdomains[189] = 22;
			elementsToSubdomains[190] = 23;
			elementsToSubdomains[191] = 23;
			#endregion

			return elementsToSubdomains;
		}

		public static Dictionary<int, int> GetSubdomainClusters()
		{
			var result = new Dictionary<int, int>();
			result[0] = 0;
			result[1] = 1;
			result[2] = 0;
			result[3] = 1;
			result[4] = 0;
			result[5] = 1;
			result[6] = 0;
			result[7] = 1;
			result[8] = 0;
			result[9] = 1;
			result[10] = 0;
			result[11] = 1;
			result[12] = 2;
			result[13] = 3;
			result[14] = 2;
			result[15] = 3;
			result[16] = 2;
			result[17] = 3;
			result[18] = 2;
			result[19] = 3;
			result[20] = 2;
			result[21] = 3;
			result[22] = 2;
			result[23] = 3;

			return result;
		}

		public static Dictionary<int, int[]> GetSubdomainNeighbors()
		{
			var result = new Dictionary<int, int[]>();
			result[0] = new int[] { 1, 2, 3, 6, 7, 8, 9, };
			result[1] = new int[] { 0, 2, 3, 6, 7, 8, 9, };
			result[2] = new int[] { 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, };
			result[3] = new int[] { 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, };
			result[4] = new int[] { 2, 3, 5, 8, 9, 10, 11, };
			result[5] = new int[] { 2, 3, 4, 8, 9, 10, 11, };

			result[6] = new int[] { 0, 1, 2, 3, 7, 8, 9, 12, 13, 14, 15, };
			result[7] = new int[] { 0, 1, 2, 3, 6, 8, 9, 12, 13, 14, 15, };
			result[8] = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, };
			result[9] = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, };
			result[10] = new int[] { 2, 3, 4, 5, 8, 9, 11, 14, 15, 16, 17, };
			result[11] = new int[] { 2, 3, 4, 5, 8, 9, 10, 14, 15, 16, 17, };

			result[12] = new int[] { 6, 7, 8, 9, 13, 14, 15, 18, 19, 20, 21, };
			result[13] = new int[] { 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, };
			result[14] = new int[] { 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, };
			result[15] = new int[] { 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, };
			result[16] = new int[] { 8, 9, 10, 11, 14, 15, 17, 20, 21, 22, 23, };
			result[17] = new int[] { 8, 9, 10, 11, 14, 15, 16, 20, 21, 22, 23, };

			result[18] = new int[] { 12, 13, 14, 15, 19, 20, 21, };
			result[19] = new int[] { 12, 13, 14, 15, 18, 20, 21, };
			result[20] = new int[] { 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, };
			result[21] = new int[] { 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, };
			result[22] = new int[] { 14, 15, 16, 17, 20, 21, 23, };
			result[23] = new int[] { 14, 15, 16, 17, 20, 21, 22, };

			return result;
		}
	}
}

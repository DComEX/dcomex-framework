.POSIX:
.SUFFIXES:
.SUFFIXES: .sh
PY = python3
PREFIX = /usr
USER = 0
D = https://download.visualstudio.microsoft.com/download/pr/01292c7c-a1ec-4957-90fc-3f6a2a1e5edc/025e84c4d9bd4aeb003d4f07b42e9159/dotnet-sdk-6.0.418-linux-x64.tar.gz

M = \
follow.py\
graph.py\
kahan.py\
pyproject.toml\
remote.py\
setup.cfg\

B = \
bin/bio\

all: lbin lib
lbin: $B
	mkdir -p -- '$(PREFIX)/bin'
	for i in $B; do cp -- "$$i" '$(PREFIX)/bin' || exit 2; done

lib: $M
	'$(PY)' -m pip install .

ldotnet:
	wget -q '$D' && \
	mkdir -p -- '$(PREFIX)/bin' && \
	tar zxf dotnet-sdk-6.0.418-linux-x64.tar.gz -C '$(PREFIX)/bin'

lmsolve:
	mkdir -p -- '$(PREFIX)/share'
	cp -- \
	msolve/CSparse.dll \
	msolve/DotNumerics.dll \
	msolve/MGroup.Constitutive.ConvectionDiffusion.dll \
	msolve/MGroup.Constitutive.Structural.dll \
	msolve/MGroup.FEM.ConvectionDiffusion.dll \
	msolve/MGroup.FEM.dll \
	msolve/MGroup.FEM.Structural.dll \
	msolve/MGroup.LinearAlgebra.Distributed.dll \
	msolve/MGroup.LinearAlgebra.dll \
	msolve/MGroup.MSolve4Korali.dll \
	msolve/MGroup.MSolve4Korali.runtimeconfig.json \
	msolve/MGroup.MSolve.Core.dll \
	msolve/MGroup.NumericalAnalyzers.Discretization.dll \
	msolve/MGroup.NumericalAnalyzers.dll \
	msolve/MGroup.Solvers.dll \
	msolve/Triangle.dll \
	msolve/RealisticMeshWithTetElements.mphtxt \
	msolve/RealisticMeshWithTetElements_t_nodes_initialCs.csv \
	msolve/RealisticMeshWithTetElements_TumorCoordinates.csv \
	'$(PREFIX)/share'

lkorali:
	cd korali && make 'USER = $(USER)' install

.sh:
	sed \
	-e 's,%mph%,"$(PREFIX)"/share/RealisticMeshWithTetElements.mphtxt,g' \
	-e 's,%csv%,"$(PREFIX)"/share/RealisticMeshWithTetElements_t_nodes_initialCs.csv,g' \
	-e 's,%tumor%,"$(PREFIX)"/share/RealisticMeshWithTetElements_TumorCoordinates.csv,g' \
	-e 's,%dll%,"$(PREFIX)"/share/MGroup.MSolve4Korali.dll,g' \
	$< > $@
	chmod a+x $@

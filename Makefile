.POSIX:
.SUFFIXES:
.SUFFIXES: .sh
PY = python3
PREFIX = /usr
USER = 0

M = \
follow.py\
graph.py\
kahan.py\
remote.py\
setup.py\

B = \
bin/bio\

all: lbin lib
lbin: $B
	mkdir -p -- "$(PREFIX)/bin"
	for i in $B; do cp -- "$$i" "$(PREFIX)/bin" || exit 2; done

lib: $M
	case '$(USER)' in \
	    0) '$(PY)' setup.py install ;; \
	    *) '$(PY)' setup.py install --user ;; \
	esac

lmsolve:
	mkdir -p -- '$(PREFIX)/share' '$(PREFIX)/bin'
	cp -- msolve/ioDir/MeshCyprusTM.mphtxt '$(PREFIX)/share'
	(cd msolve/MSolveApp/ISAAR.MSolve.MSolve4Korali && \
		DOTNET_CLI_TELEMETRY_OPTOUT=1 dotnet publish --nologo --configuration Release --output '$(PREFIX)/bin')

lkorali:
	cd korali && make 'USER = $(USER)' install

.sh:
	sed 's,%mph%,"$(PREFIX)"/share/MeshCyprusTM.mphtxt,g' $< > $@
	chmod a+x $@

clean:
	rm -rf $B korali/korali
	cd msolve/MSolveApp/ISAAR.MSolve.MSolve4Korali && \
		DOTNET_CLI_TELEMETRY_OPTOUT=1 dotnet clean --nologo

#!/bin/sh

: ${mph=%mph%}
: ${csv=%csv%}
: ${tumor=%tumor%}
: ${dll=%dll%}
: ${dotnet=dotnet}
: ${isCSparse=true}


Verbose=0 Config=0 Surrogate=0
r=$$
while :
do case "x$1" in
       x-s) Surrogate=1
	    shift
	    ;;
       x-v) Verbose=1
	    shift
	    ;;
       x-c) Config=1
	    shift
	    ;;
       x-r) shift
	    case $# in
		 0) printf >&2 'bio: error: -r needs and argumetn'
		    exit 1
		    ;;
	    esac
	    r=$1
	    shift
	    ;;
       x-h) cat >&2 <<'!'
Usage: bio [-v] [-c] [-s] [--] k1 mu sv nsteps dt [nsteps dt]

MSolve simulation of tumor growth.

Positional arguments:
  k1                   The growth rate of the tumor, 1/second.
  mu                   The shear modulus of the tumor, kPa.
  sv                   Vascular density of a tumor, m^-1.
  nsteps               Number of steps.
  dt                   The size of the timestep.

Options:
  -v                   Print verbose output during the simulation.
  -c                   Output the MSolve configuration file and exit.
  -s                   Use a surrogate instead of the full simulation.
  -r string            ID string to store output (default process ID: $$)
  -h                   Display this help message and exit.

Environment Variables (default):
  mph                  Path to the mesh of the simulated domain, in .mph format (%mph%)
  csv                  Path to the CSV with initial conditions (%csv%)
  tumor                Path to the CSV with initial tumor coordinates (%tumor%)
  isCSparse            Use sparse solver (true)
  dotnet               dotnet command (dotnet)

Returns:
  The volume of the tumor, in units of cubic millimeters.

Examples:
  bio 7.52e-11 22.44 7e3 10 1e-2
  bio 7.52e-11 22.44 7e3 5 1e-2 5 1e-1
!
	    exit 2
	    ;;
       x--) shift
	    break
	    ;;
       x-*) printf >&2 'bio: error: unknown option %s\n' "$1"
	    exit 2
	    ;;
       *) break
	  ;;
   esac
done

case $# in
    0|1|2|3|4)
	printf >&2 'bio: error: too few arguments\n'
	exit 2
	;;
    *) k1=$1; shift; mu=$1; shift; sv=$1; shift;;
esac

case $Surrogate in
    1) awk '
	BEGIN {
	  t = 0
	  for (i = 1; i < ARGC - 1; i += 2) {
	    n = ARGV[i]
	    dt = ARGV[i + 1]
	    for (j = 0; j < n; j++) {
		t += dt
		printf("%.16e %.16e\n", t, '$k1' + '$mu' * t + '$sv' * t * t)
	    }
	  }
	}' "$@"
       ;;
    0) if ! test -f "$mph"
       then printf >&2 "bio: error: mesh file '%s' is not found\n" "$mph"
	    exit 2
       fi

       if ! test -f "$csv"
       then printf >&2 "bio: error: CSV file '%s' is not found\n" "$csv"
	    exit 2
       fi

       if ! test -f "$tumor"
       then printf >&2 "bio: error: tumor coordinates file '%s' is not found\n" "$tumor"
	    exit 2
       fi

       if ! test -f "$dll"
       then printf >&2 "bio: error: DLL file '%s' is not found\n" "$dll"
	    exit 2
       fi

       if ! command >/dev/null -v "$dotnet"
       then
	   printf >&2 'bio: error: dotnet command is not avialabe\n'
	   exit 2
       fi
       config=MSolveInput.xml
       log=MSolveOutput.log
       output=MSolveOutput-x.xml
       mkdir -p "$r"
       cd -- "$r"
       # trap 'rm -f "$config" "$log" "$output"; exit 2' 1 2 3 6 14 15
       cat 2>/dev/null <<! > "$config"
<MSolve4Korali
    version="1.0">
  <Mesh>
    <MeshFile>
      $mph
    </MeshFile>
    <InitialConditionsFile>
      $csv
    </InitialConditionsFile>
    <TumorCoordinatesFile>
      $tumor
    </TumorCoordinatesFile>
  </Mesh>
  <Physics
      type="TumorGrowth"
      isCSparse="$isCSparse">
    <Timesteps>
`awk 'BEGIN {
for (i = 1; i < ARGC - 1; i += 2)
  printf "      <Timestepgroup steps=\\"%s\\" stepSizeInSeconds=\\"%s\\"/>\n", ARGV[i], ARGV[i + 1]
}' "$@"`
    </Timesteps>
  </Physics>
  <Output>
    <TumorVolume/>
  </Output>
  <Parameters>
    <k1>$k1</k1>
    <mu>$mu</mu>
    <svHost>7000</svHost>
    <svTumor>$sv</svTumor>
  </Parameters>
</MSolve4Korali>
!
       case $? in
	   0) ;;
	   *) printf >&1 "bio: fail to write to '%s'\n" "$config"
	      exit 1
	      ;;
       esac
       case $Config in
	   0) case $Verbose in
		  0) "$dotnet" "$dll" 2>/dev/null 1>/dev/null "$config" x ;;
		  1) "$dotnet" "$dll" "$config" x | tee 1>&2 "$log" ;;
	      esac
	      rc=$?
	      case $rc in
		  0) if ! test -f "$output"
		     then printf >&2 "bio: error: cannot find MSolve output '%s'\n" "$output"
			  exit 2
		     fi
	      awk -v RS='\r\n' '
	   sub(/^		<TumorVolume time="/, "") &&
	   sub(/<\/TumorVolume>$/, "") &&
	   sub(/">/, " ")' "$output"
	      ;;
		  *) echo Fail ;;
	      esac
	      ;;
	   1) cat "$config"
	      ;;
       esac
       case $Verbose in
	   0) # rm -f "$config" "$output"
	      ;;
       esac
       exit $rc
       ;;
esac

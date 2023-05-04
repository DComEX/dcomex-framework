#!/bin/sh

: ${mph=%mph%}
Verbose=0
Config=0
Surrogate=0
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
       x-h) cat >&2 <<'!'
Usage: bio [-v] [-c] [-s] k1 mu time

MSolve simulation of tumor growth.

Positional arguments:
  k1                   The growth rate of the tumor, in units of 1/second.
  mu                   The shear modulus of the tumor, in units of kPa.
  time                 The timestep to use in the simulation, in units of days.

Options:
  -v                   Print verbose output during the simulation.
  -c                   Output the MSolve configuration file and exit.
  -s                   Use a surrogate instead of the full simulation.
  -h                   Display this help message and exit.
Environment Variables:

  mph                  The path to the mesh of the simulated domain,
		       in .mph format.
Returns:
  The volume of the tumor, in units of cubic millimeters.
!
	    exit 2
	    ;;
       x-*) printf >&2 'bio: error: unknown option %s' "$1"
	    exit 2
	    ;;
       *) break
	  ;;
   esac
done

case $# in
    3) k1=$1; mu=$2; time=$3 ;;
    *) printf >&1 'bio: error: needs three arguments\n'
       exit 2
       ;;
esac

case "$time" in
    [1-9]*) ;;
    *) echo 'bio: error: time should be an positive integer' >&2; exit 2;;
esac

case $Surrogate in
    1)
	awk '
	BEGIN {
	    a = -1.4943643504045041e-07
	    b = -1.4906646615767147e-07
	    ab = 5.1441366269052818e-05
	    c = 1.4200529488187354e-08
	    d = 3.045568617688782E-6
	    x = a * '"$k1"' + b * '"$mu"' + ab * '"$k1"' * '"$mu"' + c
	    printf "%.16e\n", d * x * '"$time"'
	}'
	exit
	;;
esac


if ! test -f "$mph"
then printf "bio: error: cannot find mesh '%s'\n" "$mph" >&2
     exit 2
fi
c=/tmp/config.$$.xml
r=/tmp/result.$$.xml
trap 'rm -f $c $r; exit 2' 1 2 15
cat <<! > $c
<MSolve4Korali version="1.0">
	<Mesh>
		<File>$mph</File>
	</Mesh>
	<Physics type="TumorGrowth">
		<Time>$time</Time>
		<Timestep>1</Timestep>
	</Physics>
	<Output>
		<TumorVolume/>
	</Output>
	<Parameters>
		<k1>$k1</k1>
		<mu>$mu</mu>
	</Parameters>
</MSolve4Korali>
!

case $Config in
    0) if ! command >/dev/null -v ISAAR.MSolve.MSolve4Korali
       then
	   echo 'bio: ISAAR.MSolve.MSolve4Korali command is not avialabe' >&2
	   exit 2
       fi
       case $Verbose in
	   0) ISAAR.MSolve.MSolve4Korali 2>/dev/null 1>/dev/null $c $r ;;
	   1) ISAAR.MSolve.MSolve4Korali 1>&2 $c $r ;;
       esac
       rc=$?
       case $rc in
	   0)  if ! awk -v RS='\r\n' 'sub(/^[ \t]*<SolutionMsg>/, "") && sub(/<\/SolutionMsg>[ \t]*/, "") && /^Success$/ {exit 1}' $r
	       then
		   awk -v RS='\r\n' 'sub(/^[ \t]*<Volume>/, "") && sub(/<\/Volume>[ \t]*/, "")' $r
	       else
		   echo Fail
	       fi
	       ;;
	   *) echo Fail ;;
       esac
       ;;
    1) cat $c
       ;;
esac
case $Verbose in
    0) rm -f $c $r
       ;;
esac
exit $rc

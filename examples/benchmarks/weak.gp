set term pngcairo monochrome
set log x
set output "weak.png"
set xlabel "logical cores"
set ylabel "efficiency, %"
set key bottom left spacing 1.5
set ytics 20

plot \
"<awk '{t[$1] = $2; print $1, 100 * t[1]/$2}' bio/weak.24" u 1:2:xtic(1) w lp pt 4 ps 3 lw 3 t "draws per core: 24"

set term pngcairo monochrome
set log x
set output "strong.png"
set xlabel "logical cores"
set ylabel "efficiency, %"
set key bottom left spacing 1.5
set ytics 20

plot \
"<awk '{t[$1] = $2; print $1, 100 * t[1]/$2/$1}' bio/strong.48" u 1:2:xtic(1) w lp pt 4 ps 3 lw 3 t "draws: 48", \
"<awk '{t[$1] = $2; print $1, 100 * t[1]/$2/$1}' bio/strong.96" u 1:2:xtic(1) w lp pt 6 ps 3 lw 3 t "       96"

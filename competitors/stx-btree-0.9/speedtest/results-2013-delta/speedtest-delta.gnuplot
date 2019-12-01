#!/usr/bin/env gnuplot

set pointsize 0.7
set style line 1 linecolor rgbcolor "#FF0000" linewidth 1.6 pointsize 0.6 linetype 1
set style line 2 linecolor rgbcolor "#00FF00" linewidth 1.6 pointsize 0.6 linetype 1
set style line 3 linecolor rgbcolor "#0000FF" linewidth 1.6 pointsize 0.6 linetype 1
set style line 4 linecolor rgbcolor "#FF00FF" linewidth 1.6 pointsize 0.6 linetype 1
set style line 5 linecolor rgbcolor "#00FFFF" linewidth 1.6 pointsize 0.6 linetype 1
set style line 6 linecolor rgbcolor "#808080" linewidth 1.6 pointsize 0.6 linetype 1
set style line 7 linecolor rgbcolor "#D0D020" linewidth 1.6 pointsize 0.6 linetype 1
set style line 8 linecolor rgbcolor "#FF0000" linewidth 1.6 pointsize 0.6 linetype 2
set style line 9 linecolor rgbcolor "#00FF00" linewidth 1.6 pointsize 0.6 linetype 2
set style line 10 linecolor rgbcolor "#0000FF" linewidth 1.6 pointsize 0.6 linetype 2
set style line 11 linecolor rgbcolor "#FF00FF" linewidth 1.6 pointsize 0.6 linetype 2
set style line 12 linecolor rgbcolor "#00FFFF" linewidth 1.6 pointsize 0.6 linetype 2
set style line 13 linecolor rgbcolor "#808080" linewidth 1.6 pointsize 0.6 linetype 2
set style line 14 linecolor rgbcolor "#D0D040" linewidth 1.6 pointsize 0.6 linetype 2
set style increment user

set terminal pdf size 5, 3.5 dashed
set output 'speedtest-delta.pdf'

set label "Intel i7 920" at screen 0.04, screen 0.04

### 3rd Plot

set title "Speed Test Multimap - Normalized Time - Insertion Only (125-65536000 Items)"
set key top left
set logscale x
set xrange [100:76000000]
set xlabel "Inserts"
set ylabel "Microseconds / Insert"
set format x "%.0f"

plot "new/speed-map-insert.txt" using 1:($2 / $1) * 1000000 title "std::multimap" with linespoints, \
     "new/speed-map-insert.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multimap" with linespoints, \
     "new/speed-map-insert.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multimap<4>" with linespoints,  \
     "new/speed-map-insert.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multimap<8>" with linespoints,  \
     "new/speed-map-insert.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multimap<16>" with linespoints, \
     "new/speed-map-insert.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multimap<64>" with linespoints, \
     "new/speed-map-insert.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multimap<128>" with linespoints, \
     \
     "old/speed-map-insert.txt" using 1:($2 / $1) * 1000000 title "std::multimap" with linespoints, \
     "old/speed-map-insert.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multimap" with linespoints, \
     "old/speed-map-insert.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multimap<4>" with linespoints,  \
     "old/speed-map-insert.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multimap<8>" with linespoints,  \
     "old/speed-map-insert.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multimap<16>" with linespoints, \
     "old/speed-map-insert.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multimap<64>" with linespoints, \
     "old/speed-map-insert.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multimap<128>" with linespoints

### Now Measuring a Sequence of Insert/Find/Erase Operations

### 3rd Plot

set title "Speed Test Multimap - Normalized Time - Insert/Find/Erase (125-65536000 Items)"
set key top left
set logscale x
set xlabel "Items"
set ylabel "Microseconds / Item"
set format x "%.0f"

plot "new/speed-map-all.txt" using 1:($2 / $1) * 1000000 title "std::multimap" with linespoints, \
     "new/speed-map-all.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multimap" with linespoints, \
     "new/speed-map-all.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multimap<4>" with linespoints,  \
     "new/speed-map-all.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multimap<8>" with linespoints,  \
     "new/speed-map-all.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multimap<16>" with linespoints, \
     "new/speed-map-all.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multimap<64>" with linespoints, \
     "new/speed-map-all.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multimap<128>" with linespoints, \
     \
     "old/speed-map-all.txt" using 1:($2 / $1) * 1000000 title "std::multimap" with linespoints, \
     "old/speed-map-all.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multimap" with linespoints, \
     "old/speed-map-all.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multimap<4>" with linespoints,  \
     "old/speed-map-all.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multimap<8>" with linespoints,  \
     "old/speed-map-all.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multimap<16>" with linespoints, \
     "old/speed-map-all.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multimap<64>" with linespoints, \
     "old/speed-map-all.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multimap<128>" with linespoints

### Now Measuring only Find Operations

### 3rd Plot

set title "Speed Test Multimap - Normalized Time - Find Only (125-65536000 Items)"
set key top left
set logscale x
set xlabel "Items"
set ylabel "Microseconds / Item"
set format x "%.0f"

plot "new/speed-map-find.txt" using 1:($2 / $1) * 1000000 title "std::multimap" with linespoints, \
     "new/speed-map-find.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multimap" with linespoints, \
     "new/speed-map-find.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multimap<4>" with linespoints,  \
     "new/speed-map-find.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multimap<8>" with linespoints,  \
     "new/speed-map-find.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multimap<16>" with linespoints, \
     "new/speed-map-find.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multimap<64>" with linespoints, \
     "new/speed-map-find.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multimap<128>" with linespoints, \
     \
     "old/speed-map-find.txt" using 1:($2 / $1) * 1000000 title "std::multimap" with linespoints, \
     "old/speed-map-find.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multimap" with linespoints, \
     "old/speed-map-find.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multimap<4>" with linespoints,  \
     "old/speed-map-find.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multimap<8>" with linespoints,  \
     "old/speed-map-find.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multimap<16>" with linespoints, \
     "old/speed-map-find.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multimap<64>" with linespoints, \
     "old/speed-map-find.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multimap<128>" with linespoints




### 3rd Plot

set title "Speed Test Multiset - Normalized Time - Insertion Only (125-65536000 Items)"
set key top left
set logscale x
set xlabel "Inserts"
set ylabel "Microseconds / Insert"
set format x "%.0f"

plot "new/speed-set-insert.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "new/speed-set-insert.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multiset" with linespoints, \
     "new/speed-set-insert.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "new/speed-set-insert.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multiset<8>" with linespoints,  \
     "new/speed-set-insert.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multiset<16>" with linespoints, \
     "new/speed-set-insert.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "new/speed-set-insert.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     \
     "old/speed-set-insert.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "old/speed-set-insert.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multiset" with linespoints, \
     "old/speed-set-insert.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "old/speed-set-insert.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multiset<8>" with linespoints,  \
     "old/speed-set-insert.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multiset<16>" with linespoints, \
     "old/speed-set-insert.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "old/speed-set-insert.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints
     
### Now Measuring a Sequence of Insert/Find/Erase Operations

### 3rd Plot

set title "Speed Test Multiset - Normalized Time - Insert/Find/Erase (125-65536000 Items)"
set key top left
set logscale x
set xlabel "Items"
set ylabel "Microseconds / Item"
set format x "%.0f"

plot "new/speed-set-all.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "new/speed-set-all.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multiset" with linespoints, \
     "new/speed-set-all.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "new/speed-set-all.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multiset<8>" with linespoints,  \
     "new/speed-set-all.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multiset<16>" with linespoints, \
     "new/speed-set-all.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "new/speed-set-all.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     \
     "old/speed-set-all.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "old/speed-set-all.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multiset" with linespoints, \
     "old/speed-set-all.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "old/speed-set-all.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multiset<8>" with linespoints,  \
     "old/speed-set-all.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multiset<16>" with linespoints, \
     "old/speed-set-all.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "old/speed-set-all.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints

### Now Measuring only Find Operations

### 3rd Plot

set title "Speed Test Multiset - Normalized Time - Find Only (125-65536000 Items)"
set key top left
set logscale x
set xlabel "Items"
set ylabel "Microseconds / Item"
set format x "%.0f"

plot "new/speed-set-find.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "new/speed-set-find.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multiset" with linespoints, \
     "new/speed-set-find.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "new/speed-set-find.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multiset<8>" with linespoints,  \
     "new/speed-set-find.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multiset<16>" with linespoints, \
     "new/speed-set-find.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "new/speed-set-find.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     \
     "old/speed-set-find.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "old/speed-set-find.txt" using 1:($3 / $1) * 1000000 title "__gnu_cxx::hash_multiset" with linespoints, \
     "old/speed-set-find.txt" using 1:($5 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "old/speed-set-find.txt" using 1:($6 / $1) * 1000000 title "stx::btree_multiset<8>" with linespoints,  \
     "old/speed-set-find.txt" using 1:($7 / $1) * 1000000 title "stx::btree_multiset<16>" with linespoints, \
     "old/speed-set-find.txt" using 1:($8 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "old/speed-set-find.txt" using 1:($9 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints

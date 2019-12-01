#!/usr/bin/env gnuplot

set style line 1 linecolor rgbcolor "#FF0000" linewidth 1.6 pointsize 0.7
set style line 2 linecolor rgbcolor "#00FF00" linewidth 1.6 pointsize 0.7
set style line 3 linecolor rgbcolor "#0000FF" linewidth 1.6 pointsize 0.7
set style line 4 linecolor rgbcolor "#FF00FF" linewidth 1.6 pointsize 0.7
set style line 5 linecolor rgbcolor "#00FFFF" linewidth 1.6 pointsize 0.7
set style line 6 linecolor rgbcolor "#808080" linewidth 1.6 pointsize 0.7
set style line 7 linecolor rgbcolor "#D0D020" linewidth 1.6 pointsize 0.7
set style line 8 linecolor rgbcolor "#FF4C00" linewidth 1.6 pointsize 0.7
set style line 9 linecolor rgbcolor "#000000" linewidth 1.6 pointsize 0.7
set style increment user

set terminal pdf size 5, 3.5
set output 'speedtest.pdf'

# for generating smaller images:
# set terminal pdf size 4, 2.4

set label "Intel i7 950" at screen 0.04, screen 0.04

### 1st Plot

set title "Speed Test Multiset - Absolute Time - Insertion Only (125-32000 Items)"
set key top left
set logscale x
set xrange [100:34000]
set xlabel "Inserts"
set ylabel "Seconds"
set format x "%.0f"

plot "speed-insert.txt" using 1:2 title "std::multiset" with linespoints, \
     "speed-insert.txt" using 1:3 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-insert.txt" using 1:4 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-insert.txt" using 1:16 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-insert.txt" using 1:32 title "stx::btree_multiset<64>" with linespoints, \
     "speed-insert.txt" using 1:64 title "stx::btree_multiset<128>" with linespoints, \
     "speed-insert.txt" using 1:100 title "stx::btree_multiset<200>" with linespoints	

### 2nd Plot

set title "Speed Test Multiset - Absolute Time - Insertion Only (32000-32768000 Items)"

set xrange [30000:38000000]

replot

### 3rd Plot

set title "Speed Test Multiset - Normalized Time - Insertion Only (125-32768000 Items)"
set key top left
set logscale x
set xrange [100:38000000]
set xlabel "Inserts"
set ylabel "Microseconds / Insert"
set format x "%.0f"

plot "speed-insert.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "speed-insert.txt" using 1:($3 / $1) * 1000000 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-insert.txt" using 1:($4 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-insert.txt" using 1:($16 / $1) * 1000000 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-insert.txt" using 1:($32 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "speed-insert.txt" using 1:($64 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     "speed-insert.txt" using 1:($100 / $1) * 1000000 title "stx::btree_multiset<200>" with linespoints	

### 4th Plot

set title "Speed Test - Finding the Best Slot Size - Insertion Only - Plotted by Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"
unset logscale x
unset logscale y

plot "speed-insert.trt" using ($0*2 + 4):17 every ::2 title "8192000 Inserts" with lines, \
     "speed-insert.trt" using ($0*2 + 4):18 every ::2 title "16384000 Inserts" with lines, \
     "speed-insert.trt" using ($0*2 + 4):19 every ::2 title "32768000 Inserts" with lines

### Now Measuring a Sequence of Insert/Find/Erase Operations

### 1st Plot

set title "Speed Test Multiset - Insert/Find/Erase (125-32000 Items)"
set key top left
set logscale x
set xrange [100:34000]
set xlabel "Data Pairs"
set ylabel "Seconds"
set format x "%.0f"

plot "speed-all.txt" using 1:2 title "std::multiset" with linespoints, \
     "speed-all.txt" using 1:3 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-all.txt" using 1:4 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-all.txt" using 1:16 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-all.txt" using 1:32 title "stx::btree_multiset<64>" with linespoints, \
     "speed-all.txt" using 1:64 title "stx::btree_multiset<128>" with linespoints, \
     "speed-all.txt" using 1:100 title "stx::btree_multiset<200>" with linespoints	

### 2nd Plot

set title "Speed Test Multiset - Insert/Find/Erase (32000-32768000 Items)"

set xrange [30000:38000000]

replot

### 3rd Plot

set title "Speed Test Multiset - Normalized Time - Insert/Find/Erase (125-32768000 Items)"
set key top left
set logscale x
set xrange [100:38000000]
set xlabel "Items"
set ylabel "Microseconds / Item"
set format x "%.0f"

plot "speed-all.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "speed-all.txt" using 1:($3 / $1) * 1000000 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-all.txt" using 1:($4 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-all.txt" using 1:($16 / $1) * 1000000 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-all.txt" using 1:($32 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "speed-all.txt" using 1:($64 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     "speed-all.txt" using 1:($100 / $1) * 1000000 title "stx::btree_multiset<200>" with linespoints	

### 4th Plot

set title "Speed Test - Finding the Best Slot Size - Insert/Find/Erase - Plotted by Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"
unset logscale x
unset logscale y

plot "speed-all.trt" using ($0*2 + 4):17 every ::2 title "8192000 Data Pairs" with lines, \
     "speed-all.trt" using ($0*2 + 4):18 every ::2 title "16384000 Data Pairs" with lines, \
     "speed-all.trt" using ($0*2 + 4):19 every ::2 title "32768000 Data Pairs" with lines

### Now Measuring only Find Operations

### 1st Plot

set title "Speed Test Multiset - Find Only (125-32000 Items)"
set key top left
set logscale x
set xrange [100:34000]
set xlabel "Data Pairs"
set ylabel "Seconds"
set format x "%.0f"

plot "speed-find.txt" using 1:2 title "std::multiset" with linespoints, \
     "speed-find.txt" using 1:3 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-find.txt" using 1:4 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-find.txt" using 1:16 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-find.txt" using 1:32 title "stx::btree_multiset<64>" with linespoints, \
     "speed-find.txt" using 1:64 title "stx::btree_multiset<128>" with linespoints, \
     "speed-find.txt" using 1:100 title "stx::btree_multiset<200>" with linespoints	

### 2nd Plot

set title "Speed Test Multiset - Find Only (32000-32768000 Items)"

set xrange [30000:38000000]

replot

### 3rd Plot

set title "Speed Test Multiset - Normalized Time - Find Only (125-32768000 Items)"
set key top left
set logscale x
set xrange [100:38000000]
set xlabel "Items"
set ylabel "Microseconds / Item"
set format x "%.0f"

plot "speed-find.txt" using 1:($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "speed-find.txt" using 1:($3 / $1) * 1000000 title " __gnu_cxx::hash_multiset" with linespoints, \
     "speed-find.txt" using 1:($4 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-find.txt" using 1:($16 / $1) * 1000000 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-find.txt" using 1:($32 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "speed-find.txt" using 1:($64 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     "speed-find.txt" using 1:($100 / $1) * 1000000 title "stx::btree_multiset<200>" with linespoints	

### 4th Plot

set title "Speed Test - Finding the Best Slot Size - Find Only - Plotted by Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"
unset logscale x
unset logscale y

plot "speed-find.trt" using ($0*2 + 4):17 every ::2 title "8192000 Lookups" with lines, \
     "speed-find.trt" using ($0*2 + 4):18 every ::2 title "16384000 Lookups" with lines, \
     "speed-find.trt" using ($0*2 + 4):19 every ::2 title "32768000 Lookups" with lines

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

set label "Intel i7 920" at screen 0.04, screen 0.04

### Measuring a Sequence of Insert Operations

### 1st Plot

set title "Speed Test Multiset - Normalized Time - Insertion Only (125-65536000 Items)"
set key top left
set xrange [6.5:26.5]
set xlabel "Items [log2(n)]"
set ylabel "Microseconds / Insert"
set format x "%.0f"

plot "speed-set-insert.txt" using (log($1)/log(2)):($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "speed-set-insert.txt" using (log($1)/log(2)):($3 / $1) * 1000000 title "__gnu_cxx::hash_multiset" with linespoints, \
     "speed-set-insert.txt" using (log($1)/log(2)):($4 / $1) * 1000000 title "std::tr1::unordered_multiset" with linespoints, \
     "speed-set-insert.txt" using (log($1)/log(2)):($5 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-set-insert.txt" using (log($1)/log(2)):($19 / $1) * 1000000 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-set-insert.txt" using (log($1)/log(2)):($35 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "speed-set-insert.txt" using (log($1)/log(2)):($67 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     "speed-set-insert.txt" using (log($1)/log(2)):($103 / $1) * 1000000 title "stx::btree_multiset<200>" with linespoints

### 2nd Plot

set title "Speed Test - Finding the Best Slot Size - Insertion Only - Plotted by Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"

plot "speed-set-insert.trt" using (($0-1)*2 + 4):18 every ::2 title "16384000 Items" with lines, \
     "speed-set-insert.trt" using (($0-1)*2 + 4):19 every ::2 title "32768000 Items" with lines, \
     "speed-set-insert.trt" using (($0-1)*2 + 4):20 every ::2 title "65536000 Items" with lines

### Measuring a Sequence of Insert/Find/Erase Operations

### 1st Plot

set title "Speed Test Multiset - Normalized Time - Insert/Find/Erase (125-65536000 Items)"
set key top left
set xrange [6.5:26.5]
set xlabel "Items [log2(n)]"
set ylabel "Microseconds / Insert"
set format x "%.0f"

plot "speed-set-all.txt" using (log($1)/log(2)):($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "speed-set-all.txt" using (log($1)/log(2)):($3 / $1) * 1000000 title "__gnu_cxx::hash_multiset" with linespoints, \
     "speed-set-all.txt" using (log($1)/log(2)):($4 / $1) * 1000000 title "std::tr1::unordered_multiset" with linespoints, \
     "speed-set-all.txt" using (log($1)/log(2)):($5 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-set-all.txt" using (log($1)/log(2)):($19 / $1) * 1000000 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-set-all.txt" using (log($1)/log(2)):($35 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "speed-set-all.txt" using (log($1)/log(2)):($67 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     "speed-set-all.txt" using (log($1)/log(2)):($103 / $1) * 1000000 title "stx::btree_multiset<200>" with linespoints

### 2nd Plot

set title "Speed Test - Finding the Best Slot Size - Insert/Find/Erase - Plotted by Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"

plot "speed-set-all.trt" using (($0-1)*2 + 4):18 every ::2 title "16384000 Items" with lines, \
     "speed-set-all.trt" using (($0-1)*2 + 4):19 every ::2 title "32768000 Items" with lines, \
     "speed-set-all.trt" using (($0-1)*2 + 4):20 every ::2 title "65536000 Items" with lines

### Now Measuring only Find Operations

### 1st Plot

set title "Speed Test Multiset - Normalized Time - Find Only (125-65536000 Items)"
set key top left
set xrange [6.5:26.5]
set xlabel "Items [log2(n)]"
set ylabel "Microseconds / Insert"
set format x "%.0f"

plot "speed-set-find.txt" using (log($1)/log(2)):($2 / $1) * 1000000 title "std::multiset" with linespoints, \
     "speed-set-find.txt" using (log($1)/log(2)):($3 / $1) * 1000000 title "__gnu_cxx::hash_multiset" with linespoints, \
     "speed-set-find.txt" using (log($1)/log(2)):($4 / $1) * 1000000 title "std::tr1::unordered_multiset" with linespoints, \
     "speed-set-find.txt" using (log($1)/log(2)):($5 / $1) * 1000000 title "stx::btree_multiset<4>" with linespoints,  \
     "speed-set-find.txt" using (log($1)/log(2)):($19 / $1) * 1000000 title "stx::btree_multiset<32>" with linespoints,  \
     "speed-set-find.txt" using (log($1)/log(2)):($35 / $1) * 1000000 title "stx::btree_multiset<64>" with linespoints, \
     "speed-set-find.txt" using (log($1)/log(2)):($67 / $1) * 1000000 title "stx::btree_multiset<128>" with linespoints, \
     "speed-set-find.txt" using (log($1)/log(2)):($103 / $1) * 1000000 title "stx::btree_multiset<200>" with linespoints

### 2nd Plot

set title "Speed Test - Finding the Best Slot Size - Find Only - Plotted by Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"

plot "speed-set-find.trt" using (($0-1)*2 + 4):18 every ::2 title "16384000 Items" with lines, \
     "speed-set-find.trt" using (($0-1)*2 + 4):19 every ::2 title "32768000 Items" with lines, \
     "speed-set-find.trt" using (($0-1)*2 + 4):20 every ::2 title "65536000 Items" with lines

################################################################################

### Measuring a Sequence of Insert Operations

### 1st Plot

set title "Speed Test Multimap - Normalized Time - Insertion Only (125-65536000 Items)"
set key top left
set xrange [6.5:26.5]
set xlabel "Items [log2(n)]"
set ylabel "Microseconds / Insert"
set format x "%.0f"

plot "speed-map-insert.txt" using (log($1)/log(2)):($2 / $1) * 1000000 title "std::multimap" with linespoints, \
     "speed-map-insert.txt" using (log($1)/log(2)):($3 / $1) * 1000000 title "__gnu_cxx::hash_multimap" with linespoints, \
     "speed-map-insert.txt" using (log($1)/log(2)):($4 / $1) * 1000000 title "std::tr1::unordered_multimap" with linespoints, \
     "speed-map-insert.txt" using (log($1)/log(2)):($5 / $1) * 1000000 title "stx::btree_multimap<4>" with linespoints,  \
     "speed-map-insert.txt" using (log($1)/log(2)):($19 / $1) * 1000000 title "stx::btree_multimap<32>" with linespoints,  \
     "speed-map-insert.txt" using (log($1)/log(2)):($35 / $1) * 1000000 title "stx::btree_multimap<64>" with linespoints, \
     "speed-map-insert.txt" using (log($1)/log(2)):($67 / $1) * 1000000 title "stx::btree_multimap<128>" with linespoints, \
     "speed-map-insert.txt" using (log($1)/log(2)):($103 / $1) * 1000000 title "stx::btree_multimap<200>" with linespoints

### 2nd Plot

set title "Speed Test - Finding the Best Slot Size - Insertion Only - Plotted by Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"

plot "speed-map-insert.trt" using (($0-1)*2 + 4):18 every ::2 title "16384000 Items" with lines, \
     "speed-map-insert.trt" using (($0-1)*2 + 4):19 every ::2 title "32768000 Items" with lines, \
     "speed-map-insert.trt" using (($0-1)*2 + 4):20 every ::2 title "65536000 Items" with lines

### Measuring a Sequence of Insert/Find/Erase Operations

### 1st Plot

set title "Speed Test Multimap - Normalized Time - Insert/Find/Erase (125-65536000 Items)"
set key top left
set xrange [6.5:26.5]
set xlabel "Items [log2(n)]"
set ylabel "Microseconds / Insert"
set format x "%.0f"

plot "speed-map-all.txt" using (log($1)/log(2)):($2 / $1) * 1000000 title "std::multimap" with linespoints, \
     "speed-map-all.txt" using (log($1)/log(2)):($3 / $1) * 1000000 title "__gnu_cxx::hash_multimap" with linespoints, \
     "speed-map-all.txt" using (log($1)/log(2)):($4 / $1) * 1000000 title "std::tr1::unordered_multimap" with linespoints, \
     "speed-map-all.txt" using (log($1)/log(2)):($5 / $1) * 1000000 title "stx::btree_multimap<4>" with linespoints,  \
     "speed-map-all.txt" using (log($1)/log(2)):($19 / $1) * 1000000 title "stx::btree_multimap<32>" with linespoints,  \
     "speed-map-all.txt" using (log($1)/log(2)):($35 / $1) * 1000000 title "stx::btree_multimap<64>" with linespoints, \
     "speed-map-all.txt" using (log($1)/log(2)):($67 / $1) * 1000000 title "stx::btree_multimap<128>" with linespoints, \
     "speed-map-all.txt" using (log($1)/log(2)):($103 / $1) * 1000000 title "stx::btree_multimap<200>" with linespoints

### 2nd Plot

set title "Speed Test - Finding the Best Slot Size - Insert/Find/Erase - Plotted by Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"

plot "speed-map-all.trt" using (($0-1)*2 + 4):18 every ::2 title "16384000 Items" with lines, \
     "speed-map-all.trt" using (($0-1)*2 + 4):19 every ::2 title "32768000 Items" with lines, \
     "speed-map-all.trt" using (($0-1)*2 + 4):20 every ::2 title "65536000 Items" with lines

### Now Measuring only Find Operations

### 1st Plot

set title "Speed Test Multimap - Normalized Time - Find Only (125-65536000 Items)"
set key top left
set xrange [6.5:26.5]
set xlabel "Items [log2(n)]"
set ylabel "Microseconds / Insert"
set format x "%.0f"

plot "speed-map-find.txt" using (log($1)/log(2)):($2 / $1) * 1000000 title "std::multimap" with linespoints, \
     "speed-map-find.txt" using (log($1)/log(2)):($3 / $1) * 1000000 title "__gnu_cxx::hash_multimap" with linespoints, \
     "speed-map-find.txt" using (log($1)/log(2)):($4 / $1) * 1000000 title "std::tr1::unordered_multimap" with linespoints, \
     "speed-map-find.txt" using (log($1)/log(2)):($5 / $1) * 1000000 title "stx::btree_multimap<4>" with linespoints,  \
     "speed-map-find.txt" using (log($1)/log(2)):($19 / $1) * 1000000 title "stx::btree_multimap<32>" with linespoints,  \
     "speed-map-find.txt" using (log($1)/log(2)):($35 / $1) * 1000000 title "stx::btree_multimap<64>" with linespoints, \
     "speed-map-find.txt" using (log($1)/log(2)):($67 / $1) * 1000000 title "stx::btree_multimap<128>" with linespoints, \
     "speed-map-find.txt" using (log($1)/log(2)):($103 / $1) * 1000000 title "stx::btree_multimap<200>" with linespoints

### 2nd Plot

set title "Speed Test - Finding the Best Slot Size - Find Only - Plotted by Slots in B+ Tree"

set key top right
set autoscale x
set xlabel "Leaf/Inner Slots"
set ylabel "Seconds"

plot "speed-map-find.trt" using (($0-1)*2 + 4):18 every ::2 title "16384000 Items" with lines, \
     "speed-map-find.trt" using (($0-1)*2 + 4):19 every ::2 title "32768000 Items" with lines, \
     "speed-map-find.trt" using (($0-1)*2 + 4):20 every ::2 title "65536000 Items" with lines

#!/usr/bin/gawk -f
# Transpose a matrix: assumes all lines have same number
# of fields

NR == 1 {
    n = NF
    for (i = 2; i <= NF; i++)
	row[i] = $i
    next
}

{
    if (NF > n)
	n = NF
    for (i = 2; i <= NF; i++)
	row[i] = row[i] " " $i
}

END {
    for (i = 2; i <= n; i++)
	print row[i]
}


mkdir -p ./results_join

#for method in {0..7}
#do
#    for sel in {0..2}
#    do
#        sync ; echo 3 > /proc/sys/vm/drop_caches
#        build/join $method 0 $sel | tee results_join/${method}_s${sel}.txt
#    done
#done

# warmup
build/join 7 1 0

for method in {0..7}
do
    for sel in {0..2}
    do
        build/join $method 1 $sel | tee results_join/${method}_s${sel}p.txt
    done
done

              
cat results_join/*.txt > results_join/all.txt
 
chmod -R 777 results_join

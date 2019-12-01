rm -f ../competitors/rmi/all_rmis.h

for dataset in $(cat ../scripts/datasets_under_test.txt); do
    echo "#include \"${dataset}_rmi.h\"" >> ../competitors/rmi/all_rmis.h
done

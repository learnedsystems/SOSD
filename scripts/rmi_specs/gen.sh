rm -f competitors/rmi/all_rmis.h

echo "#pragma once" >> competitors/rmi/all_rmis.h
for header in $(ls competitors/rmi/ | grep "\\.h$" | grep -v data | grep -v all_rmis ); do
    echo "#include \"${header}\"" >> competitors/rmi/all_rmis.h
done

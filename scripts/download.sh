
# Calculate md5 checksum of FILE and stores it in MD5_RESULT
function get_checksum() {
   FILE=$1

   if [ -x "$(command -v md5sum)" ]; then
      # Linux
      MD5_RESULT=`md5sum ${FILE} | awk '{ print $1 }'`
   else
      # OS X
      MD5_RESULT=`md5 -q ${FILE}`
   fi
}


function download_file_zst() {
   FILE=$1;
   CHECKSUM=$2;
   URL=$3;

   # Check if file already exists
   if [ -f ${FILE} ]; then
      # Exists -> check the checksum
      get_checksum ${FILE}
      if [ "${MD5_RESULT}" != "${CHECKSUM}" ]; then
         wget -O - ${URL} | zstd -d > ${FILE}
      fi
   else
      # Does not exists -> download
      wget -O - ${URL} | zstd -d > ${FILE}
   fi

   # Validate (at this point the file should really exist)
   get_checksum ${FILE}
   if [ "${MD5_RESULT}" != "${CHECKSUM}" ]; then
      echo "error checksum does not match: run download again"
      exit -1
   else
      echo ${FILE} "checksum ok"
   fi
}

# Main script code
function main() {
   echo "downloading data ..."
   mkdir -p data
   cd data

   # Format: download_file <file_name> <md5_checksum> <url>
   download_file wiki_ts_200M_uint64 4f1402b1c476d67f77d2da4955432f7d https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/ULSWQQ 
   download_file osm_cellids_200M_uint64 01666e42b2d64a55411bdc280ac9d2a3  https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/CTBUKT 

   download_file books_200M_uint32 c4a848fdc56130abdd167d7e6b813843 https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/MZZUP2 
   download_file books_200M_uint64 aa88040624be2f508f1ab6f5532ace88 https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/PDOUMU

   download_file fb_200M_uint32 881eacb62c38eb8c2fdd4d59706b70a7 https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/GXH3ZC 
   download_file fb_200M_uint64 407a23758b72e3c1ee3f6384d98ce604 https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/Y54SI9 
   cd ..
   echo "done"
}

# Run
main


echo "Generating synthetic data..."
python3 gen_uniform.py --many --uint32
python3 gen_uniform.py --many
python3 gen_uniform.py --many --sparse --uint32
python3 gen_uniform.py --many --sparse
python3 gen_norm.py


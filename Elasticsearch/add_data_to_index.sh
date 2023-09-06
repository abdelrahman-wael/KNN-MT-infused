# !/bin/bash

echo -e "Running ELASTIC script \n\n"


if [ -z "$1" ];
    then ip=localhost;
    else ip=$1;
fi

if [ -z "$2" ];
    then port=9200;
    else port=$2;
fi

if [ -z "$3" ]; then
    echo -e "Index name is not provided! Exiting..\n\n"
    exit 1
else 
    index_name=$(echo $3|tr '[:upper:]' '[:lower:]');
fi

if [ -z "$4" ]; then
    echo -e "Data path is not passed!! \n\n"
    exit 1
else 
    data_path=$4;
fi

if [ ! -f "$data_path" ]; then
    echo -e "Data file path: '${4}' does not exist! -- Exiting!"
    exit 1
fi

echo -e "Uging ip: $ip : port as $port\n\n"
echo -e "Index name is: $index_name"

echo -e "\nStep 1: GET elastic cluster health"
curl -sk http://$ip:$port/_cluster/health?pretty


echo -e "\nStep 2: Adding data to index: ${index_name}"
echo -e "Time for indexing file: ${data_path}:"
time curl -s -XPUT "http://${ip}:${port}/${index_name}/_bulk?filter_path=took,errors" -H 'Content-Type: application/json' --data-binary "@${data_path}"


# echo -e "\nStep 3: GET elastic current indecies info"
# curl -sk -XGET http://$ip:$port/_cat/indices?v

# curl -sk -XGET http://10.32.160.10:40001/_nodes/_all/jvm?pretty
# curl -sk -XGET http://10.32.160.10:40001/_cluster/health?pretty

# ES_JAVA_OPTS="-Xms2g -Xmx2g" ./bin/elasticsearch
# ./assign_data_to_index.sh 10.32.160.10 40001 test_index_1 tmp/data.json
#!/bin/bash

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

echo -e "Uging ip: $ip : port as $port\n\n"
echo -e "Index name is: $index_name"

echo -e "\nStep 1: GET elastic cluster health\n\n"
curl -sk http://$ip:$port/_cluster/health?pretty

echo -e "\nStep 2: Adding index with name: ${index_name} \n\n"
curl -sv -XPUT "http://${ip}:${port}/${index_name}?pretty"  -H "Content-Type: application/json" -d '
{
"mappings": {
    "properties": {"default_field": {"type": "text"}}
            },
"settings": {
    "index": {
      "number_of_shards": 2,  
      "number_of_replicas": 1 
    }
  }
}
'

echo -e "\nStep 3: GET elastic current indecies info\n\n"
curl -sk -XGET http://$ip:$port/_cat/indices?v
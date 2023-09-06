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

echo -e "Uging ip: $ip and port: $port\n\n"


echo -e "\nStep 1: GET elastic is running\n\n"
curl -sk http://$ip:$port?pretty

echo -e "\nStep 2: GET elastic cluster health\n\n"
curl -sk http://$ip:$port/_cluster/health?pretty

echo -e "\nStep 3: GET elastic current indecies info\n\n"
curl -sk -XGET http://$ip:$port/_cat/indices?v

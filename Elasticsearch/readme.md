# Elasticsearch usage


## Install and run Elasticsearch

```shell
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.5.2-linux-x86_64.tar.gz

wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.5.2-linux-x86_64.tar.gz.sha512

tar -xzf elasticsearch-8.5.2-linux-x86_64.tar.gz

cd elasticsearch-8.5.2/ 

./bin/elasticsearch
```
## First time run
- You will see credintials printed, you may save the password as you might need it later.

## Modify the config yaml
- Ctrl + c the working server
- Open config/elasticsearch.yaml

- Make these changes:
``` yaml
    http.port: 40001
    # Enable security features
    xpack.security.enabled: false

    xpack.security.enrollment.enabled: true

    # Enable encryption for HTTP API client connections, such as Kibana, Logstash, and Agents
    xpack.security.http.ssl:
    enabled: false
    keystore.path: certs/http.p12

    # Enable encryption and mutual authentication between cluster nodes
    xpack.security.transport.ssl:
    enabled: false
    verification_mode: certificate
    keystore.path: certs/transport.p12
    truststore.path: certs/transport.p12
```

- and add this:
```yaml
    http.max_content_length: 1800mb
```

- Install `screen`
```shell
    sudo apt update
    sudo apt install screen
    screen --version
```

- Create screen: `screen -S elastic`

- Run the elastic server on it: `./bin/elasticsearch`


## Send health check request

- From the server running machine get the ip address (ie. 10.20.30.40)
```shell
    hostname -i
```

- In earlier step we configured http.port to be 40001 - you may change it.

- Health check request is as follows:

```curl
    curl -sk http://$ip:$port/_cluster/health?pretty
```
OR
- Use the bash here for more cluster info
```bash
    ./check_health.sh ip port
```


## Create index

**NOTE index name must be all lower cases**

- Run the following bash file

```bash
    ./create_sim_index.sh ip port index_name
```


## Index data

- Run the follwing python script: `add_data_to_index.py`

- type `python add_data_to_index.py -h` for manual help.

```python
python add_data_to_index.py -x ${source_file} -y ${target_file} -o tmp_out_file  -log logging_file -l SrcTgt -i index_name -ip ip -p port
```

## Retrieve data

- Import function call: `search_data` from the file `search_data`

- Call it in your code, expecting the resulting output to be a dict.

- Where keys is 'test_sent_i' and value is a list of data_stores

- Example:
```python
    results = search_data(index_name="fren_index_1m", lang_dir='fren',
                    test_file="/mnt/mainz01eus/amrhendy/projects/zcode_vnext/gpt3.5/data/authtest_2205/fren/test/test.fr-en.fr",
                    es_size=64, store_size=32, outputs_path="tmp", log_path="log", ip_address="10.32.160.10", port="40001")
```

- Another version of this is to be used for storing data stores

- Simply, add the following argument: `save_outs` to be one of these ["elastic", "rerank", "both", None]
default is None

- Files will be saved at: `{out_dir}/{index_name}_results_{size}/test_sample_{i}`

- Terminal output:
    - ELASTICSEARCH API SAYS HELLO !!!
    - Source is: fr    Target is: en
    - Timing: Argument parsing. time: 0.1339 seconds
    - Timing: Reading files. time: 0.0115 seconds
    - Timing: Processed files. time: 0.1053 seconds
    - Timing: CURL only time: 105.4452 seconds
    - Timing: re-ranking only time: 5.5888 seconds
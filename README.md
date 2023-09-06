# Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started
1.	Setting up envirnoment
    - currently we use this yml file to start Singularity job `/sing_non_prem_v1.yml`
    - clone the repo and checkout to branch `ababouelenin/interactive_demo`
        - if you are using the yml file above, just go to `~/CT/KNN-MT` and checkout to branch `ababouelenin/interactive_demo`
    - then SSH on the running job, and install the following 
        - ```pip install unbabel-comet``` this is for Comet evaluation
2.	Starting the ElasticSearch engine
    - Install and run Elasticsearch
    ```shell
    wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.5.2-linux-x86_64.tar.gz

    wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.5.2-linux-x86_64.tar.gz.sha512

    tar -xzf elasticsearch-8.5.2-linux-x86_64.tar.gz

    cd elasticsearch-8.5.2/ 

    ./bin/elasticsearch
    ```
    - Ctrl + c the working server
    - Open config/elasticsearch.yaml
    - Make these changes:
    ``` yaml
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
        http.port: 40001
    ```
    - Then rerun the elastic server using: `./bin/elasticsearch`

3.	Starting the Backend (written in Flask)
    - go to the root path of the project
    - run `bash start_demo.sh`
4.	Using the System and hitting the endpoints
    - ### Endpoint "/index"
        - Creates index for the data provided in the elasticsearch engine
        - Output a domain ID that can be used in translation later on
        - example body 
        ```json 
        {
            "source_data": "/mnt/mtcairogroup/Users/mohamedmaher/CTv2/customer_bugs/DeMT-Data/en-cs-medpharma/train/train/en-cs.medpharma.train.enu.snt",
            "target_data": "/mnt/mtcairogroup/Users/mohamedmaher/CTv2/customer_bugs/DeMT-Data/en-cs-medpharma/train/train/en-cs.medpharma.train.csy.snt",
            "lang_dir": "encs",
            "domain_id": "medpharma_"
        }
        ```
    - ### Endpoint "/translate"
        - Translate a list of sentences and augment the MT model output with the KNN distribution if the use_knn flag is enabled and following domain specified in the `domain_id`
        - Output List of translation
        - example body 
        ```json 
        {
            "document": [
                "Das Fahrzeug lädt anschließend bis zur unteren Ladegrenze."
            ],
            "lang_dir": "deen",
            "use_knn": true,
            "batch_size":20,
            "domain_id": [
                "defryjxhk"
            ],
        }
        ```
    - ### Endpoint "/translate_file"
        - Translate a file and augment the MT model output with the KNN distribution if the use_knn flag is enabled and following domain specified in the `domain_id`, returning a list of translation
        - Output List of translation
        - example body 
        ```json 
        {
        "document": src,
        "lang_dir": lang_dir,
        "batch_size": 4,
        "use_knn": use_knn,
        "domain_id": [domain_id],
        }
        ```
    
# Hosting the Demo Frontend 
1.	Setting up envirnoment
    - Install Node.js
    - run `npm install` inside the knn-demo directory
2.	Starting the frontend server
    - run `npm start` inside the knn-demo directory
    - the frontend assumes that the Backend is running locally on `http://localhost:12346/`
    - the frontend will be hosted automatically on `http://localhost:3000/`


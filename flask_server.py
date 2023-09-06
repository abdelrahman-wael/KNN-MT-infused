from flask import Flask, request, jsonify
import sys
import json
import subprocess
from Elasticsearch.search_data import search_data
from Elasticsearch.add_data_to_index import add_data_to_index
import shutil
import os
import random
import string
from fairseq_cli.demo import KNNRetrieval

from flask_cors import CORS, cross_origin

application = Flask(__name__)
cors = CORS(application)
application.config['CORS_HEADERS'] = 'Content-Type'


ip="localhost"
e_port="40001"

def read_files(src_file,tgt_file=None):
            src_file_pointer = open(src_file,"r",encoding="utf-8")
            src_lines = src_file_pointer.readlines()
            if tgt_file:
                tgt_file_pointer=open(tgt_file,"r",encoding="utf-8")
                tgt_lines = tgt_file_pointer.readlines()
                return src_lines , tgt_lines
            return src_lines

def get_datastore_from_files(files_location, list_of_sentences, src, tgt):
    
    data = {}

    for i in range(len(list_of_sentences)):
        # open file 
        

        source = []
        target = []
        test_line = []

        f = open(os.path.join(files_location, "test_sample_"+str(i)+".json"))
        ds = json.load(f)
        
        source = []
        target = []

        for ds_item in ds:
            source.append(ds_item[src])
            target.append(ds_item[tgt])

        if len(source) != len(target):
            raise Exception(len(source), len(target), " sizes are not equal")

        if len(source) == 0:
            empty_datastores += [filename]
            source = ["0"]
            target = ["0"]

        data[list_of_sentences[i]] = [{"source": s, "target": t} for s, t in zip(source, target)]
        # print(list_of_sentences[i])
    
    return data
        

@application.route("/index", methods=["POST"])
@cross_origin()
#define function
def new_domain():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
    else:
        return 'Content-Type not supported!'
    # import pdb;pdb.set_trace()
    source_data = json["source_data"]
    lang_dir = json["lang_dir"]
    domain_name = json["domain_id"]
    domain_id = str(domain_name)+ "".join(random.choices(string.ascii_lowercase, k=5))
    print("created new domain id", domain_id)


    result = subprocess.run(["./Elasticsearch/create_sim_index.sh", ip, e_port, domain_id], stdout=subprocess.PIPE)

    os.chdir('./Elasticsearch')

    add_data_to_index(
            source_file = json["source_data"],
            target_file = json["target_data"],
            outputs_path = "outputs", log_path = "logs", 
            index_name = domain_id, address = ip, port = e_port, lang_dir = lang_dir
    )
    os.chdir('../')
    # 
    # 
    #####
    
    return {"domain_id":domain_id}

@application.route("/translate", methods=["POST"])
@cross_origin()
#define function
def translate():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
    else:
        return 'Content-Type not supported!'


    list_of_sentences = json["document"]
    target_sents=json["target"]
    domain_id = json["domain_id"][0]
    use_KNN=json["use_knn"]
    interactive_bsz=json["batch_size"]
    lang_dir=json["lang_dir"]

    takeDsFromFiles = False

    if "elasticSearchLocation" in json:
        elasticSearchLocation=json["elasticSearchLocation"]
        takeDsFromFiles = True
 
    if use_KNN:
        if takeDsFromFiles:
            print("using datastore from files")
            datastore = get_datastore_from_files(elasticSearchLocation, list_of_sentences)
        else:
            print("creating datastore ..")
            os.chdir('./Elasticsearch')
            datastore = search_data(
                index_name=domain_id,
                lang_dir=lang_dir,
                in_src=list_of_sentences,
                es_size=64,
                store_size=15,
                out_dir="/home/aiscuser/tmp_ds",
                log_path="log",
                ip_address=ip,
                save_outs=None,
                port="40001")
            os.chdir('../')

        knn.translate(list_of_sentences,datastore,use_KNN=True,interactive_bsz=int(interactive_bsz),english_centric=json["english_centric"],target_sents=target_sents)
    else:
        knn.translate(list_of_sentences,{},use_KNN=False,interactive_bsz=int(interactive_bsz),english_centric=json["english_centric"],target_sents=target_sents)

    return knn.trans_results.results



@application.route("/translate_file", methods=["POST"])
@cross_origin()
def translate_file():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
    else:
        return 'Content-Type not supported!'


    src_path = json["document"]
    tgt_path=json["target"]
    list_of_sentences = read_files(src_path)
    target_sents=read_files(tgt_path)
    domain_id = json["domain_id"][0]
    use_KNN=json["use_knn"]
    interactive_bsz=json["batch_size"]
    lang_dir=json["lang_dir"]
    source_lang=json["source_lang"]
    target_lang=json["target_lang"]
    takeDsFromFiles = False

    if "elasticSearchLocation" in json:
        elasticSearchLocation=json["elasticSearchLocation"]
        takeDsFromFiles = True
 
    if use_KNN:
        if takeDsFromFiles:
            print("using datastore from files")
            datastore = get_datastore_from_files(elasticSearchLocation, list_of_sentences, lang_dir[:2],  lang_dir[2:])
        else:
            print("creating datastore ..")
            os.chdir('/home/azureuser/cloudfiles/code/Users/abdelrahman.abouelenin/KNN-MT/Elasticsearch')
            # import pdb;pdb.set_trace()
            datastore = search_data(
                index_name=domain_id,
                lang_dir=lang_dir,
                in_src=list_of_sentences,
                es_size=64,
                store_size=15,
                out_dir="/home/azureuser/cloudfiles/code/Users/abdelrahman.abouelenin/indices/tmp_ds",
                log_path="log",
                ip_address=ip,
                save_outs=None,
                port="40001")
            os.chdir('../')
        # import pdb;pdb.set_trace()
        knn.translate(list_of_sentences,datastore,use_KNN=True,interactive_bsz=int(interactive_bsz),source_lang=source_lang,target_lang=target_lang,target_sents=target_sents)
    else:
        knn.translate(list_of_sentences,{},use_KNN=False,interactive_bsz=int(interactive_bsz),source_lang=source_lang,target_lang=target_lang,target_sents=target_sents)


    return [knn.trans_results.results]


def Dummy_translate():

    list_of_sentences=["Fahrrad-Lift","Fahrrad-Seitenstaender"]
    datastore={}
    # datastore["Fahrrad-Lift"]=[{"source": "Fahrrad-Seitenstaender", "target": "Bike side stands", "score": 0.4545454545454546}]
    # datastore["Fahrrad-Seitenstaender"]=[{"source": "Fahrrad-Seitenstaender", "target": "Bike side stands", "score": 0.4545454545454546}]

    knn.translate(list_of_sentences,datastore,use_KNN=False)
    print(knn.trans_results.results)


knn = KNNRetrieval()
# Dummy_translate()

if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except:
        port = 12346 
# model_path=
# dictionary=
# demo(model_path,dictionary)


application.run(port=port, debug=False)


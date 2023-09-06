from flask import Flask, request, jsonify
import sys
import subprocess

application = Flask(__name__)

@application.route("/index", methods=["POST"])
#define function
def new_domain():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
    else:
        return 'Content-Type not supported!'

    data = json.sentences
    #####
    # Create index given data
    # - return Customer id
    #####
    
    return model["model"]

@application.route("/translate", methods=["POST"])
#define function
def translate():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
    else:
        return 'Content-Type not supported!'


    list_of_sentences = json["document"]
    #####
    # save json file with 15 sent for each item in list_of_sentences
    # - return link like this https://mainz01eus.blob.core.windows.net/user/v-enarouz/data/ELASTICSEARCH/de_en/data_store_de_en_2500_test_elastic_search_and_edit_distance_top_64_15/
    #######


    file1 = open('/home/mohamedmaher/document_to_translate.de', 'w')
    file1.writelines(json["document"])
    file1.close()

    P = subprocess.Popen(['sh', "/mnt/mtcairogroup/Users/mohamedmaher/code/run_sequential_elastic_knn_adaptive_batched--index.sh"])
    P.wait()
    # rc = call(  )

    file1 = open('/home/mohamedmaher/translated_documnet', 'r')
    output = file1.readlines()
    file1.close()

    return {"translation": output}
    # return "hello world"



if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except:
        port = 12345 

application.run(port=port, debug=True)
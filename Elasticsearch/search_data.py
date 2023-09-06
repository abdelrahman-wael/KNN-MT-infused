'''
Here we parse the inputs X-Y into the json that ElasticSearch expects
'''
import argparse
import os
import gzip
import time
import subprocess
import json
import Levenshtein

PIDS = set()

def _fetch_tsv_or_gz_file(file_path):

    if file_path.split(".")[-1] == "gz":
        in_file = gzip.open(file_path, 'rt',encoding="utf8")

    #"tsv" or other
    else:
        in_file = open(file_path, encoding="utf8")

    return in_file


def time_this(tic, msg):
    print(f"[search_data.py]: Timing: {msg} time: {time.perf_counter() - tic:0.4f} seconds")
    return time.perf_counter()


def replace_for_json(sent):
    sent = sent.replace('"', "'")
    sent = sent.replace("\\", "")
    sent = sent.replace("\n", "")
    return sent


def curl_search_this(file_path, index_name, ip_address, port, id, log_path):
    # print(f"Started indixing the {id}th file...")
    subprocess.call(f"bash ./retrieve_data_from_index.sh {ip_address} {port} {index_name} {file_path} > {log_path}/search_{id}.log",
                        executable='/bin/bash', shell=True)
    # PIDS.add(p.pid)



def lev(sent1, sent2):
    
    if len(sent1) == 0 and len(sent2) ==0: return 0
    score = (1- (Levenshtein.distance(sent1,sent2)/max(len(sent1),len(sent2))) )
    return score




def main_lev(source, list_of_sources, m, src_lang, tgt_lang):

    scores = {}
    for i,es_source in enumerate(list_of_sources):
        scores[i] = {
            'source':es_source[src_lang],
            'target':es_source[tgt_lang],
            'score':lev(source, es_source[src_lang])
        }

    scores_sorted = dict(sorted(scores.items(),key=lambda x:x[1]['score'],reverse = True))
    # print(scores_sorted)
    results = []
    for i,(_,sample) in enumerate(scores_sorted.items()):
        if i < m:
            results.append(sample)

    # print(results)
    return results


def search_data(**kwargs):
    print(f"[search_data.py]: ELASTICSEARCH API SAYS HELLO !!!")

    tic = time.perf_counter()

    in_src      = kwargs["in_src"]
    out_dir     = kwargs["out_dir"]
    lang_dir    = kwargs["lang_dir"]
    index_name  = kwargs["index_name"]
    es_size     = kwargs["es_size"]
    store_size  = kwargs["store_size"]
    batch_size  = kwargs["batch_size"]  if "batch_size" in kwargs.keys() else None #not used yet!
    log_path    = kwargs["log_path"]
    ip_address  = kwargs["ip_address"]
    port        = kwargs["port"]
    log_rate    = kwargs["log_rate"]    if "log_rate"   in kwargs.keys() else None #not used yet!
    parallel    = kwargs["parallel"]    if "parallel"   in kwargs.keys() else None #not used yet!
    save_outs   = kwargs["save_outs"]   if "save_outs"  in kwargs.keys() else None # need def to be None
    
    
    # TODO: batching is not affective currently, as test sets we see is retavely small.

    # assert batch_size >= store_size, "Batch size can't be less than data store size"
    # assert batch_size%8 ==0 and store_size%8==0, "Batch size and Data sotre size need to be divisible by 8"
    # test_sent_limit = batch_size / store_size

    # assert os.path.exists(in_src), "Input test file does not exist!"


    if not os.path.exists(out_dir):
        print(f"[search_data.py]: Warning: Output directory does not exist, will create it at: {out_dir}")
        os.mkdir(out_dir)
    
    if not os.path.exists(log_path):
        print(f"[search_data.py]: Warning: Log directory does not exist, will create it at: {log_path}")
        os.mkdir(log_path)
    
    if not os.path.exists(f"{out_dir}/{index_name}_results_{es_size}"):
        print(f"[search_data.py]: Warning: ES results will be stored at: {out_dir}/{index_name}_results_{es_size}")
        os.mkdir(f"{out_dir}/{index_name}_results_{es_size}")

    if not os.path.exists(f"{out_dir}/{index_name}_results_{store_size}"):
        print(f"[search_data.py]: Warning: Reranking results will be stored at: {out_dir}/{index_name}_results_{store_size}")
        os.mkdir(f"{out_dir}/{index_name}_results_{store_size}")
    
    # extract source and target langs
    src_lang, tgt_lang = lang_dir[:2], lang_dir[2:]
    # tgt_lang='fr'
    print(f"[search_data.py]: Source is: {src_lang} \t Target is: {tgt_lang}")

    tic = time_this(tic, "Argument parsing.")

    # read source and target files
    # src_lines = _fetch_tsv_or_gz_file(in_src).read().splitlines()
    src_lines = in_src

    tic = time_this(tic, "Reading files.")

    # index_line = '{"index": "' + index_name + '"}'
    index_line = '{ }'

    shard_i = -1
    L = len(src_lines)
    counter = 0
    while counter <= L-1:
        shard_i += 1
        output_path = f"{out_dir}/test_file_{shard_i}.json"

        with open(output_path, "w", encoding="utf8") as f_out:
            for i, src_line in enumerate(src_lines):
                counter += 1

                src_line = replace_for_json(src_line)  
                    
                doc_line = '{"size":' + str(es_size) + ',"query":{"match":{"' + src_lang + '":"' + src_line + '"}},"_source":["' +src_lang+'","'+tgt_lang+'"]}'

                f_out.write(index_line)
                f_out.write("\n")
                f_out.write(doc_line)
                f_out.write("\n")
                
                # TODO: For now we assume it's only one file
                # if it exceeds size limit, in most cases it won't

                # if i%10_000 == 0:
                #     f_out.seek(0, os.SEEK_END)
                #     # 2_147_483_647
                #     if f_out.tell() + 200_000_000 >= 1_887_436_800:
                #         src_lines = src_lines[i+1:]
                #         break
            
            tic = time_this(tic, "Processed files.")

        # once a file is out, call the search api
        curl_search_this(file_path=output_path, index_name=index_name, ip_address=ip_address, port=port, id=shard_i, log_path=log_path) 

    
    # TODO: when parralising .. probably we won't
    # print(f"Current pids running: {PIDS}")
    # while len(PIDS) != 0:
    #     try:
    #         pid, _ = os.wait()
    #         if pid in PIDS:
    #             # print(pid, "finished")
    #             PIDS.remove(pid)
    #     except:
    #         break

    tic = time_this(tic, "CURL only")

    # Now you can apply the re-ranking loop
    output_path = f"{out_dir}/test_file_{shard_i}.json"
    results = {}

    # TODO: they might be multiple files, if you have 5M+ test sentences!
    with open(f"{log_path}/search_0.log", encoding="utf-8") as f:
        data = json.load(f)

        responses = data["responses"]

        for i, (src_line, res) in enumerate(zip(src_lines,responses)):
            sources = []
            sources_hit = []
            for hit in res['hits']['hits']:
                sources.append(hit["_source"])
                if save_outs == "elastic" or save_outs == "both":
                    hit_w_score = hit["_source"]
                    hit_w_score["_score"] = hit["_score"]
                    sources_hit.append(hit_w_score)

            if save_outs == "elastic" or save_outs == "both":
                with open(f"{out_dir}/{index_name}_results_{es_size}/test_sample_{i}.ds", "w", encoding="utf8") as f:
                    for line_print in sources_hit:
                        f.write(str(line_print))
                        f.write("\n")

            top_G = main_lev(src_line, sources, store_size, src_lang, tgt_lang)

            results[src_line] = top_G

            if save_outs == "rerank" or save_outs == "both":
                with open(f"{out_dir}/{index_name}_results_{store_size}/test_sample_{i}.ds", "w", encoding="utf8") as f:
                    for line in top_G:
                        f.write(str(line))
                        f.write("\n")

    tic = time_this(tic, "re-ranking only")

    return results


if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-x', '--in_src',       help='path to test file', type=str, required=True)
        parser.add_argument('-o', '--out_dir',      help='path to output files to be stored at.', type=str, required=True)
        parser.add_argument('-l', '--lang_dir',     help='language direction (ie. enfr, fren...)', type=str, required=True)
        parser.add_argument('-i', '--index_name',   help='you need to specify an index name for that operation!', type=str, required=True)
        parser.add_argument('-s', '--es_size',      help='elasticsearch retrieved size', type=int, required=True)
        parser.add_argument('-d', '--store_size',   help='data store size', type=int, required=True)
        parser.add_argument('-b', '--batch_size',   help='batch size required', type=int, required=False, default=1024)
        parser.add_argument('-log', '--log_path',   help='path to log files', type=str, required=True)
        parser.add_argument('-ip', '--ip_address',  help='host address to which elastic will respond to, (ie. 10.20.30.40)', type=str, required=True)
        parser.add_argument('-p', '--port',         help='port gate of the address', type=str, required=True)
        parser.add_argument('-r', '--log_rate',     help='log rate of the lines processed', type=int, required=False, default=100000)
        parser.add_argument('-ll', '--parallel',    help='enable this only when you confident about cluster settings', type=bool, required=False, default=False)
        parser.add_argument('-v', '--save_outs',    help='Which files would be saved; ["elastic" ,"rerank", "both", None]', type=str, required=True)
        
        args = parser.parse_args()

        in_src      = args.in_src
        out_dir     = args.out_dir
        lang_dir    = args.lang_dir
        index_name  = args.index_name
        es_size     = args.es_size
        store_size  = args.store_size
        batch_size  = args.batch_size
        log_path    = args.log_path
        ip_address  = args.ip_address
        port        = args.port
        log_rate    = args.log_rate
        parallel    = args.parallel
        save_outs   = args.save_outs

        search_data(
            in_src      = in_src,
            out_dir     = out_dir,
            lang_dir    = lang_dir,
            index_name  = index_name,
            es_size     = es_size,
            store_size  = store_size,
            batch_size  = batch_size,
            log_path    = log_path,
            ip_address  = ip_address,
            port        = port,
            log_rate    = log_rate,
            parallel    = parallel,
            save_outs   = save_outs,                  
            )

    except:
        print("[search_data.py]: Warning: args are not passed appropriately, hopefully you're on the func call mode.")



'''

results = search_data(index_name="fren_index_1m", lang_dir='fren',
                    test_file="/mnt/mainz01eus/amrhendy/projects/zcode_vnext/gpt3.5/data/authtest_2205/fren/test/test.fr-en.fr",
                    es_size=64, store_size=32, out_dir="tmp", log_path="log", ip_address="10.32.160.10", port="40001", save_outs=None)


python search_data.py   -x /mnt/mainz01eus/amrhendy/projects/zcode_vnext/gpt3.5/data/authtest_2205/fren/test/test.fr-en.fr \
                        -o tmp -l fren -i fren_index_1m -s 64 -d 32 -log log -ip 10.32.160.10 -p 40001
'''

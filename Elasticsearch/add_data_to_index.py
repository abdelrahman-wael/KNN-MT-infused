'''
Here we parse the inputs X-Y into the json that ElasticSearch expects
and send request using cURL to index the ready files.
'''
import argparse
import os
import gzip
import time
import subprocess

PIDS = set()

def _fetch_tsv_or_gz_file(file_path):

    if file_path.split(".")[-1] == "gz":
        in_file = gzip.open(file_path, 'rt',encoding="utf8")

    #"tsv" or other
    else:
        in_file = open(file_path, encoding="utf8")

    return in_file


def time_this(tic, msg):
    print(f"Timing: {msg} time: {time.perf_counter() - tic:0.4f} seconds")
    return time.perf_counter()


def replace_for_json(sent):
    sent = sent.replace('"', "'")

    sent = sent.replace("\\", "")

    sent = sent.replace("\n", "")
    return sent



def curl_index_this(file_path, index_name, ip, port, id, log_path):
    print(f"Started indixing the {id}th file...")
    p = subprocess.Popen(f"bash ./add_data_to_index.sh {ip} {port} {index_name} {file_path} > {log_path}/index_{id}.log",executable='/bin/bash', shell=True)
    PIDS.add(p.pid)
    



# if __name__ == "__main__":

def add_data_to_index(**kwargs):
    print(f"ELASTICSEARCH API SAYS HELLO !!!")

    tic = time.perf_counter()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-x', '--source_file',  help='path to source file', type=str, required=True)
    # parser.add_argument('-y', '--target_file',  help='path to target file', type=str, required=True)
    # parser.add_argument('-o', '--outputs_path', help='path to output files', type=str, required=True)
    # parser.add_argument('-l', '--lang_dir',     help='language direction (ie. enfr, fren...)', type=str, required=True)
    # parser.add_argument('-i', '--index_name',   help='you need to specify an index name for that operation!', type=str, required=True)
    # parser.add_argument('-r', '--log_rate',     help='log rate of the lines processed', type=int, required=False, default=100000)
    # parser.add_argument('-log', '--log_path',   help='path to log files', type=str, required=True)
    # parser.add_argument('-ip', '--address',      help='host address to which elastic will respond to, (ie. 10.20.30.40)', type=str, required=True)
    # parser.add_argument('-p', '--port',         help='port gate of the address', type=str, required=True)
    # parser.add_argument('-ll', '--parallel',    help='enable this only when you confident about cluster settings', type=bool, required=False, default=False)



    # args = parser.parse_args()

    # in_src = args.source_file
    # log_rate = args.log_rate
    # index_name = args.index_name.lower()
    # in_tgt = args.target_file
    # ip = args.address
    # port = args.port
    # out_dir = args.outputs_path
    # log_path = args.log_path
    # parallel = args.parallel
    
    in_src      = kwargs["source_file"]
    in_tgt      = kwargs["target_file"]
    out_dir     = kwargs["outputs_path"]
    log_rate    = kwargs["log_rate"]            if "log_rate" in kwargs.keys() else 100000
    index_name  = kwargs["index_name"].lower()
    ip          = kwargs["address"]
    port        = kwargs["port"]
    log_path    = kwargs["log_path"]
    parallel    = kwargs["parallel"]            if "parallel" in kwargs.keys() else False
    lang_dir    = kwargs["lang_dir"]

    assert os.path.exists(in_src) or os.path.exists(in_tgt), "Input directory does not exist!"


    if not os.path.exists(out_dir):
        print(f"Warning: Output directory does not exist, will create it at: {out_dir}")
        os.mkdir(out_dir)

    if not os.path.exists(log_path):
        print(f"Warning: log_path directory does not exist, will create it at: '{log_path}'")
        os.mkdir(log_path)

    # extract source and target langs
    src_lang, tgt_lang = lang_dir[:2], lang_dir[2:]
    print(f"Source is: {src_lang} \t Target is: {tgt_lang}")

    tic = time_this(tic, "Argument parsing.")

    # read source and target files
    # src_lines = in_src
    src_lines = _fetch_tsv_or_gz_file(in_src).readlines()
    # tgt_lines = in_tgt
    tgt_lines = _fetch_tsv_or_gz_file(in_tgt).readlines()
    assert len(src_lines) == len(tgt_lines), "Length of source lines does not match target lines..."
    
    tic = time_this(tic, "Reading files.")

    index_line = '{"index": {"_index":"' + index_name + '"}}'
    shard_i = -1
    L = len(src_lines)
    counter = 0
    while counter <= L-1:
        shard_i += 1
        output_path = f"{out_dir}/bulk_file_{shard_i}.json"

        with open(output_path, "w", encoding="utf8") as f_out:
            for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
                counter += 1

                src_line = replace_for_json(src_line)
                tgt_line = replace_for_json(tgt_line)
                
                doc_line = '{"' + src_lang + '":"'+ src_line + '","'+ tgt_lang + '":"'+ tgt_line + '"}'

                f_out.write(index_line)
                f_out.write("\n")
                f_out.write(doc_line)
                f_out.write("\n")
            
                if i%10000 == 0:
                    f_out.seek(0, os.SEEK_END)
                    # 2_147_483_647
                    if f_out.tell() + 200_000_000 >= 1_887_436_800:
                        src_lines = src_lines[i+1:]
                        tgt_lines = tgt_lines[i+1:]
                        break

        # call the curl to index the ready files
        if parallel:
            curl_index_this(file_path=output_path, index_name=index_name, ip=ip, port=port, id=shard_i, log_path=log_path) 
            print(f"Indexing process is placed..")


    
        tic = time_this(tic, f"Batch {shard_i} at line: {counter}.")


    # make sure all subprocess are finished before exiting
    if parallel:
        print(f"Current pids running: {PIDS}")
        while len(PIDS) != 0:
            pid, _ = os.wait()
            if pid in PIDS:
                print(pid, "finished")
                PIDS.remove(pid)
    
    else:
        # then sequentially index all files at out_dir
        bulk_files = os.listdir(out_dir)
        for i, bulk_file in enumerate(bulk_files):
            
            output_path = f"{out_dir}/{bulk_file}"
            curl_index_this(file_path=output_path, index_name=index_name, ip=ip, port=port, id=i, log_path=log_path) 
            print(f"Indexing file: {bulk_file}")

            while len(PIDS) != 0:
                pid, _ = os.wait()
                if pid in PIDS:
                    PIDS.remove(pid)            




'''
python add_data_to_index.py -x /mnt/mainz01eus/amrhendy/projects/zcode_vnext/gpt3.5/data/polonium_data_pool/fren/top_20M/train.fr-en.fr \
                            -y /mnt/mainz01eus/amrhendy/projects/zcode_vnext/gpt3.5/data/polonium_data_pool/fren/top_20M/train.fr-en.en \
                            -o tmp_2  -log logs -l fren -i index_test_2 -a 10.32.160.10 -p 40001
    add_data_to_index(
            source_file = "/mnt/mainz01eus/amrhendy/projects/zcode_vnext/gpt3.5/data/polonium_data_pool/fren/top_20M/train.fr-en.fr",
            target_file = "/mnt/mainz01eus/amrhendy/projects/zcode_vnext/gpt3.5/data/polonium_data_pool/fren/top_20M/train.fr-en.en",
            outputs_path = "outpus", log_path = "logs", 
            index_name = "index_test_2", address = "10.32.160.10", port = "40001", lang_dir = "fren"
    )
'''


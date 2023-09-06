from search_data import search_data

inds = ["ec_deen_20k"]
for index_name in inds:
    search_data(
        in_src      = "/mnt/mainz01eus/v-enarouz/data/deen_ecommerce/ende/test/test.en-de.de",
        out_dir     = f"out_{index_name}_15",
        lang_dir    = "deen",
        index_name  = index_name,
        es_size     = 64,
        store_size  = 15,
        batch_size  = None,
        log_path    = f"log_{index_name}",
        ip_address  = "10.32.160.10",
        port        = "40001",
        log_rate    = None,
        parallel    = False,
        save_outs   = "rerank",                  
        )


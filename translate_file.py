import requests
import json 



url = 'http://127.0.0.1:12346/translate_file'
# url = 'http://127.0.0.1:12346/index'


# myobj={
#     "source_data": "/home/azureuser/cloudfiles/code/Users/abdelrahman.abouelenin/data/domain_data/emea_de_fr_en_data/defr/emea.train.de-fr.de",
#     "target_data": "/home/azureuser/cloudfiles/code/Users/abdelrahman.abouelenin/data/domain_data/emea_de_fr_en_data/defr/emea.train.de-fr.fr",
#     "lang_dir": "defr",
#     "domain_id": "emea"
# }

# "document": "/mnt/mtcairogroup/Users/mohamedmaher/CTv2/customer_bugs/DeMT-Data/en-cs-medpharma/test/en-cs.medpharma.test.enu.snt",

# test deen 

#  /mnt/mainz01eus/ababouelenin/projects/XY_CT/data/test/deen/test.de-en.de

# emea
myobj = {
    "document": "/home/azureuser/cloudfiles/code/Users/abdelrahman.abouelenin/data/domain_data/deen/law/400_subset/test.de",
    "target":"/home/azureuser/cloudfiles/code/Users/abdelrahman.abouelenin/data/domain_data/deen/law/400_subset/test.en",
    "lang_dir": "deen",
    "use_knn":True,
    "batch_size":"1",
    "english_centric":False,
    "source_lang":"de",
    "target_lang":"en",
    "domain_id": [
        "lawcgopf",
        "emeaeskdc",#emea_defr
        "vw_1msdnls"
    ]
}

# law

# myobj = {
#     "document": "/home/azureuser/cloudfiles/code/Users/abdelrahman.abouelenin/data/domain_data/deen/Users/abdelrahman.abouelenin/data/domain_data/deen/law/test_subset_200.denen",
#     "use_knn":True,
#     "batch_size":"200",
#     "english_centric":False,
#     "source_lang":"de",
#     "target_lang":"en",
#     "domain_id": [
#         "lawcgopf",
#         "emeaeskdc",
#         "vw_1msdnls"
#     ]
# }







import time
start = time.time()
x = requests.post(url, json = myobj).json()[0]
end = time.time()
# print(x)
print("time of http request, " + str(end-start))
# print(myobj["document"] + "   "+myobj["use_KNN"] +"   "+myobj["interactive_bsz"])
print("done")
# 
file1 = open('/home/azureuser/cloudfiles/code/Users/abdelrahman.abouelenin/data/domain_data/deen/law/400_subset/only_pretrained_t100.de-en.en', 'w')
file1.writelines([l+'\n' for l in x])
file1.close()


# cat /home/aiscuser/output_vw/without_knn_output_bsz_20_finetuned.txt | sacrebleu /mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/data/CT_vw/wv_ct_data/vw/ende/test/test.en-de.en
# comet-score -s /mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/data/CT_vw/wv_ct_data/vw/ende/test/test.en-de.de -t /home/aiscuser/output_vw/without_knn_output_bsz_20_finetuned.txt -r /mnt/mainz01eus/ababouelenin/projects/CustomTranslation/debugging_node/knnmt/blob_data/data/CT_vw/wv_ct_data/vw/ende/test/test.en-de.en --gpus 8 --model "wmt20-comet-da"

#cat /home/aiscuser/taus_gpu_index_fix/knn_output_bsz_4.txt | sacrebleu /mnt/mainz01eus/projects/KNNMT_CT/data/taus_ecommerce/test/ecommerce.en-de.test.enu.snt 
# comet-score -s /mnt/mainz01eus/projects/KNNMT_CT/data/taus_ecommerce/test/ecommerce.en-de.test.deu.snt -t /home/aiscuser/output_taus_ecommerce/without_knn_output_bsz_20_finetuned.txt -r /mnt/mainz01eus/projects/KNNMT_CT/data/taus_ecommerce/test/ecommerce.en-de.test.enu.snt --gpus 8 --model "wmt20-comet-da"
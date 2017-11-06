####################################
# Author: Shashi Narayan
# Date: September 2016
# Project: Document Summarization
# H2020 Summa Project
####################################

"""
Document Summarization Modules and Models
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
import re
import os.path

from pyrouge import Rouge155
import json
from multiprocessing import Pool

from my_flags import FLAGS

def _rouge(system_dir, gold_dir):
    # Run rouge
    r = Rouge155()
    r.system_dir = system_dir
    r.model_dir = gold_dir
    r.system_filename_pattern = '([a-zA-Z0-9]*).model'
    r.model_filename_pattern = '#ID#.gold'
    output = r.convert_and_evaluate(rouge_args="-e Code/neuralsum/ROUGE_evaluation/rouge/data -a -c 95 -m -n 4 -w 1.2")
    # print output
    output_dict = r.output_to_dict(output)
    # print output_dict
    
    avg_rscore = 0
    
    avg_rscore = (output_dict["rouge_1_recall"]+output_dict["rouge_2_recall"]+
                  output_dict["rouge_3_recall"]+output_dict["rouge_4_recall"]+
                  output_dict["rouge_l_recall"])/5.0
    return avg_rscore
               
def _rouge_wrapper_traindata(docname, final_labels):
    # Gold Summary Directory : Always use original sentences
    gold_summary_directory = FLAGS.gold_summary_directory + "/gold-"+FLAGS.data_mode+"-training-org"
    gold_summary_fileaddress = gold_summary_directory + "/" + docname + ".gold"

    # Prepare Gold Model File
    os.system("mkdir -p "+FLAGS.tmp_directory+"/gold-"+docname)
    os.system("cp "+gold_summary_fileaddress+" "+FLAGS.tmp_directory+"/gold-"+docname+"/")
    
    # Document Sentence: Always use original sentences to generate summaries
    doc_sent_fileaddress = FLAGS.doc_sentence_directory + "/" + FLAGS.data_mode + "/training-sent/"+docname+".summary.final.org_sents"
    doc_sents = open(doc_sent_fileaddress).readlines()

    # Prepare Model file
    os.system("mkdir -p "+FLAGS.tmp_directory+"/model-"+docname)
    
    # SHASHI: FIX THIS
    labels_ones = [idx for idx in range(len(final_labels[:len(doc_sents)])) if final_labels[idx]=="1"]
    model_highlights = [doc_sents[idx] for idx in labels_ones]
    foutput = open(FLAGS.tmp_directory+"/model-"+docname+"/"+docname+".model" , "w")
    foutput.write("".join(model_highlights))
    foutput.close()

    return _rouge(FLAGS.tmp_directory+"/model-"+docname, FLAGS.tmp_directory+"/gold-"+docname)

def _multi_run_wrapper(args):
    return _rouge_wrapper_traindata(*args)

class Reward_Generator:
    def __init__(self):
        self.rouge_dict = {}
        
        # Start a pool
        self.pool = Pool(5)

    def save_rouge_dict(self):
        with open(FLAGS.train_dir+"/rouge-dict.json", 'w') as outfile:
            json.dump(self.rouge_dict, outfile)
            
    def restore_rouge_dict(self):
        self.rouge_dict = {}
        if os.path.isfile(FLAGS.train_dir+"/rouge-dict.json"):
            with open(FLAGS.train_dir+"/rouge-dict.json") as data_file: 
                self.rouge_dict = json.load(data_file)

    def get_full_rouge(self, system_dir, datatype):
        # Gold Directory: Always use original files
        gold_summary_directory = FLAGS.gold_summary_directory + "/gold-"+FLAGS.data_mode+"-"+datatype+"-org"

        rouge_score = _rouge(system_dir, gold_summary_directory)

        # Delete any tmp file
        os.system("rm -r "+FLAGS.tmp_directory+"/tmp*")

        return rouge_score
        
    def get_batch_rouge(self, batch_docnames, batch_predicted_labels):
        
        # Numpy dtype
        dtype = np.float16 if FLAGS.use_fp16 else np.float32
        
        # Batch Size
        batch_size = len(batch_docnames)

        # batch_rouge 
        batch_rouge = np.empty(batch_size, dtype=dtype)
        
        # Estimate list of arguments to run pool
        didx_list = []
        docname_labels_list = []
        for docindex in range(batch_size):
            docname = batch_docnames[docindex] 
            predicted_labels = batch_predicted_labels[docindex]
            
            # Prepare final labels for summary generation
            final_labels = [str(int(predicted_labels[sentidx][0])) for sentidx in range(FLAGS.max_doc_length)]
            # print(final_labels)

            isfound = False
            rougescore = 0.0
            if docname in self.rouge_dict:
                final_labels_string = "".join(final_labels)
                if final_labels_string in self.rouge_dict[docname]:
                    rougescore = self.rouge_dict[docname][final_labels_string]
                    isfound = True

            if isfound:
                # Update batch_rouge
                batch_rouge[docindex] = rougescore
            else:
                didx_list.append(docindex)
                docname_labels_list.append((docname, final_labels))
        
        # Run parallel pool
        if(len(didx_list) > 0):
            # Run in parallel
            rougescore_list = self.pool.map(_multi_run_wrapper,docname_labels_list)
            # Process results
            for didx, rougescore, docname_labels in zip(didx_list, rougescore_list, docname_labels_list):
                # Update batch_rouge
                batch_rouge[didx] = rougescore
                
                # Update rouge dict
                docname = docname_labels[0]
                final_labels_string = "".join(docname_labels[1])
                if docname not in self.rouge_dict:
                    self.rouge_dict[docname] = {final_labels_string:rougescore}
                else:
                    self.rouge_dict[docname][final_labels_string] = rougescore
            # Delete any tmp file
            os.system("rm -r "+ FLAGS.tmp_directory+"/tmp* " + FLAGS.tmp_directory+"/gold-* " + FLAGS.tmp_directory+"/model-*")
        # print(self.rouge_dict)
        return batch_rouge
        

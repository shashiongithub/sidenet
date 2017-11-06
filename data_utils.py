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

from my_flags import FLAGS
from model_utils import convert_logits_to_softmax, predict_toprankedthree

# Special IDs
PAD_ID = 0
UNK_ID = 1

class Data:
    def __init__(self, vocab_dict, data_type):
        self.filenames = []
        self.docs = []
        self.titles = []
        self.images = []
        self.firstsentences = []
        self.randomsentences = []
        self.labels = []
        self.weights = []

        self.fileindices = []

        self.data_type = data_type

        # populate the data 
        self.populate_data(vocab_dict, data_type)
        
        # Write to files
        self.write_to_files(data_type)
        
    def write_prediction_summaries(self, pred_logits, modelname, session=None):
        print("Writing predictions and final summaries ...")

        # Convert to softmax logits
        pred_logits = convert_logits_to_softmax(pred_logits, session=session)
        # Save Output Logits
        np.save(FLAGS.train_dir+"/"+modelname+"."+self.data_type+"-prediction", pred_logits)

        # Writing
        pred_labels = predict_toprankedthree(pred_logits, self.weights)
        self.write_predictions(modelname+"."+self.data_type, pred_logits, pred_labels)
        self.process_predictions_rankedtopthree(modelname+"."+self.data_type)

    def write_predictions(self, file_prefix, np_predictions, np_labels):
        foutput = open(FLAGS.train_dir+"/"+file_prefix+".predictions", "w")
        for fileindex in self.fileindices:
            filename = self.filenames[fileindex]
            foutput.write(filename+"\n")
            
            # print(filename) 
            # print(np_predictions[fileindex])
            # print(np_labels[fileindex])

            sentcount = 0
            for sentpred, sentlabel in zip(np_predictions[fileindex], np_labels[fileindex]):
                one_prob = sentpred[0]
                label = sentlabel[0]

                if self.weights[fileindex][sentcount] == 1: 
                    foutput.write(str(int(label))+"\t"+str(one_prob)+"\n")
                else:
                    break
  
                sentcount += 1
            foutput.write("\n")
        foutput.close()

    def process_predictions_rankedtopthree(self, file_prefix):
        predictiondata = open(FLAGS.train_dir+"/"+file_prefix+".predictions").read().strip().split("\n\n")
        # print len(predictiondata)
        
        summary_dirname = FLAGS.train_dir+"/"+file_prefix+"-summary-rankedtop3"
        os.system("mkdir "+summary_dirname)
        
        for item in predictiondata:
            # print(item)
            
            itemdata = item.strip().split("\n")
            # print len(itemdata)
            
            filename = itemdata[0]
            # print filename
            
            # predictions file already have top three sentences marked 
            final_sentids = []
            for sentid in range(len(itemdata[1:])):
                label_score = itemdata[sentid+1].split()
                if label_score[0] == "1":
                    final_sentids.append(sentid)

            # Create final summary files
            fileid = filename.split("/")[-1][:-14] # .summary.final
            summary_file = open(summary_dirname+"/"+fileid+".model", "w")

            # Read Sents in the document : Always use original sentences
            sent_filename = FLAGS.doc_sentence_directory + "/" + FLAGS.data_mode + "/"+self.data_type+"-sent/"+fileid+".summary.final.org_sents"
            docsents = open(sent_filename).readlines()

            # Top Ranked three sentences 
            selected_sents = [docsents[sentid] for sentid in final_sentids if sentid < len(docsents)]
            # print(selected_sents)

            summary_file.write("".join(selected_sents)+"\n")
            summary_file.close()
        
    # def process_predictions_all1(self, file_prefix):
    #     predictiondata = open(FLAGS.train_dir+"/"+file_prefix+".predictions").read().strip().split("\n\n")
    #     # print len(predictiondata)
        
    #     summary_dirname_all1 = FLAGS.train_dir+"/"+file_prefix+"-summary-all1"
    #     os.system("mkdir "+summary_dirname_all1)
        
    #     for item in predictiondata:
    #         itemdata = item.strip().split("\n")
    #         # print len(itemdata)
            
    #         filename = itemdata[0]
    #         # print filename
            
    #         sentid_ones = []
    #         for sentid in range(len(itemdata[1:])):
    #             # print sentid,  itemdata[sentid+1]
    #             label_score = itemdata[sentid+1].split()
    #             if label_score[0] == "1":
    #                 sentid_ones.append(sentid)

    #         # Create final summary files
    #         fileid = filename.split("/")[-1][:-14] # .summary.final
                    
    #         summary_file_all1 = open(summary_dirname_all1+"/"+fileid+".model", "w")
    #         # Read Sents in the document
    #         sent_filename = ""
    #         if (FLAGS.anonymized_setting):
    #             sent_filename = FLAGS.doc_sentence_directory + "/" + FLAGS.data_mode + "/"+self.data_type+"-sent/"+fileid+".summary.final.anonym_sents" 
    #         else:
    #             sent_filename = FLAGS.doc_sentence_directory + "/" + FLAGS.data_mode + "/"+self.data_type+"-sent/"+fileid+".summary.final.org_sents"
            
    #         docsents = open(sent_filename).readlines()
    #         # All 1
    #         # Get selected sentences
    #         selected_sents = [docsents[sentid] for sentid in sentid_ones] #[:3]]
    #         summary_file_all1.write("".join(selected_sents)+"\n")
    #         summary_file_all1.close()
        
    # def process_predictions_onestopthree(self, file_prefix):
    #     predictiondata = open(FLAGS.train_dir+"/"+file_prefix+".predictions").read().strip().split("\n\n")
    #     # print len(predictiondata)
        
    #     summary_dirname_top1 = FLAGS.train_dir+"/"+file_prefix+"-summary-top1"
    #     os.system("mkdir "+summary_dirname_top1)
        
    #     for item in predictiondata:
    #         itemdata = item.strip().split("\n")
    #         # print len(itemdata)
            
    #         filename = itemdata[0]
    #         # print filename
            
    #         sentid_ones = []
    #         for sentid in range(len(itemdata[1:])):
    #             # print sentid,  itemdata[sentid+1]
    #             label_score = itemdata[sentid+1].split()
    #             if label_score[0] == "1":
    #                 sentid_ones.append(sentid)

    #         # Create final summary files
    #         fileid = filename.split("/")[-1][:-14] # .summary.final 
                    
    #         summary_file_top1 = open(summary_dirname_top1+"/"+fileid+".model", "w")
    #         # Read Sents in the document
    #         sent_filename = ""
    #         if (FLAGS.anonymized_setting):
    #             sent_filename = filename+".anonym_sents"
    #         else:
    #             sent_filename = filename+".org_sents"
            
    #         docsents = open(sent_filename).read().strip().split("\n")
    #         # Top 1
    #         # Get selected sentences
    #         selected_sents = [docsents[sentid] for sentid in sentid_ones[:3]]
    #         summary_file_top1.write("\n".join(selected_sents)+"\n")
    #         summary_file_top1.close()

    # def get_labels_weights(self):

    #     # Numpy dtype
    #     dtype = np.float16 if FLAGS.use_fp16 else np.float32

    #     all_label = np.empty((len(self.fileindices), FLAGS.max_doc_length, FLAGS.target_label_size), dtype=dtype)
    #     all_weight = np.empty((len(self.fileindices), FLAGS.max_doc_length), dtype=dtype)
        
    #     batch_idx = 0
    #     for fileindex in self.fileindices:
    #         # Labels
    #         labels = self.labels[fileindex]
    #         # labels: (max_doc_length) --> labels_vecs: (max_doc_length, target_label_size)
    #         labels_vecs = [[1, 0] if (label==1) else [0, 1] for label in labels]
    #         all_label[batch_idx] = np.array(labels_vecs[:], dtype=dtype)

    #         # Weights
    #         weights = self.weights[fileindex]
    #         all_weight[batch_idx] = np.array(weights[:], dtype=dtype)
            
    #         # increase batch count
    #         batch_idx += 1

    #     return all_label, all_weight

    def get_batch(self, startidx, endidx): 
        # This is very fast if you keep everything in Numpy
        
        # Numpy dtype
        dtype = np.float16 if FLAGS.use_fp16 else np.float32
        
        # For train, (endidx-startidx)=FLAGS.batch_size, for others its as specified
        batch_docnames = np.empty((endidx-startidx), dtype="S40") # File ID of size 40
        batch_docs = np.empty(((endidx-startidx), (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length + 
                                                   FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.max_sent_length), dtype="int32") 
        batch_label = np.empty(((endidx-startidx), FLAGS.max_doc_length, FLAGS.target_label_size), dtype=dtype) 
        batch_weight = np.empty(((endidx-startidx), FLAGS.max_doc_length), dtype=dtype) 
        
        batch_idx = 0
        for fileindex in self.fileindices[startidx:endidx]:
            # Document Names
            batch_docnames[batch_idx] = self.filenames[fileindex][67:-14]

            # Document
            doc_wordids = self.docs[fileindex][:] # [FLAGS.max_doc_length, FLAGS.max_sent_length]
            if (FLAGS.max_title_length > 0):
                doc_wordids = doc_wordids + self.titles[fileindex][:] # [FLAGS.max_title_length, FLAGS.max_sent_length]
            if (FLAGS.max_image_length > 0):
                doc_wordids = doc_wordids + self.images[fileindex][:] # [FLAGS.max_image_length, FLAGS.max_sent_length]    
            if (FLAGS.max_firstsentences_length > 0):
                doc_wordids = doc_wordids + self.firstsentences[fileindex][:] # [FLAGS.max_firstsentences_length, FLAGS.max_sent_length]
            if (FLAGS.max_randomsentences_length > 0):
                doc_wordids = doc_wordids + self.randomsentences[fileindex][:] # [FLAGS.max_randomsentences_length, FLAGS.max_sent_length]

            batch_docs[batch_idx] = np.array(doc_wordids[:], dtype="int32")
            
            # Labels
            labels = self.labels[fileindex]
            # labels: (max_doc_length) --> labels_vecs: (max_doc_length, target_label_size)
            labels_vecs = [[1, 0] if (label==1) else [0, 1] for label in labels]
            batch_label[batch_idx] = np.array(labels_vecs[:], dtype=dtype)

            # Weights
            weights = self.weights[fileindex]
            batch_weight[batch_idx] = np.array(weights[:], dtype=dtype)
            
            # increase batch count
            batch_idx += 1

        return batch_docnames, batch_docs, batch_label, batch_weight

    def shuffle_fileindices(self):
        random.shuffle(self.fileindices)

    def write_to_files(self, data_type):
        full_data_file_prefix = ""
        if FLAGS.anonymized_setting:
            full_data_file_prefix = FLAGS.train_dir + "/" + FLAGS.data_mode + "-" + data_type+".anonym_ent"
        else:
            full_data_file_prefix = FLAGS.train_dir + "/" + FLAGS.data_mode + "-" + data_type+".org_ent"
        print("Writing data files with prefix (.filename, .doc, .title, .image, .label, .weight): %s"%full_data_file_prefix)

        ffilenames = open(full_data_file_prefix+".filename", "w")
        fdoc = open(full_data_file_prefix+".doc", "w")
        ftitle = open(full_data_file_prefix+".title", "w")
        fimage = open(full_data_file_prefix+".image", "w")
        ffirst = open(full_data_file_prefix+".first", "w")
        frandom = open(full_data_file_prefix+".random", "w")
        flabel = open(full_data_file_prefix+".label", "w")
        # flabel = open(full_data_file_prefix+".label-mod", "w")
        # flabel = open(full_data_file_prefix+".label-oracle", "w")
        fweight = open(full_data_file_prefix+".weight", "w")

        for filename, doc, title, image, first, random, label, weight in zip(self.filenames, self.docs, self.titles, self.images, 
                                                                             self.firstsentences, self.randomsentences, self.labels, self.weights):
            ffilenames.write(filename+"\n")
            fdoc.write("\n".join([" ".join([str(item) for item in itemlist]) for itemlist in doc])+"\n\n")
            ftitle.write("\n".join([" ".join([str(item) for item in itemlist]) for itemlist in title])+"\n\n")
            fimage.write("\n".join([" ".join([str(item) for item in itemlist]) for itemlist in image])+"\n\n")
            ffirst.write("\n".join([" ".join([str(item) for item in itemlist]) for itemlist in first])+"\n\n")
            frandom.write("\n".join([" ".join([str(item) for item in itemlist]) for itemlist in random])+"\n\n")
            flabel.write(" ".join([str(item) for item in label])+"\n")
            fweight.write(" ".join([str(item) for item in weight])+"\n")
        ffilenames.close()
        fdoc.close()
        ftitle.close()
        fimage.close()
        ffirst.close()
        frandom.close()
        flabel.close()
        fweight.close()

    def populate_data(self, vocab_dict, data_type):
        
        def process_to_chop_pad(orgids, requiredsize):
            if (len(orgids) >= requiredsize):
                return orgids[:requiredsize]
            else:
                padids = [PAD_ID] * (requiredsize - len(orgids))
                return (orgids + padids)
        

        full_data_file_prefix = ""
        if FLAGS.anonymized_setting:
            full_data_file_prefix = FLAGS.preprocessed_data_directory + "/" + FLAGS.data_mode + "/" + data_type+".anonym_ent"
        else:
            full_data_file_prefix = FLAGS.preprocessed_data_directory + "/" + FLAGS.data_mode + "/" + data_type+".org_ent"

        print("Data file prefix (.doc, .title, .image, .label.jp-org): %s"%full_data_file_prefix)
        
        # Process doc, title, image and label
        doc_data_list = open(full_data_file_prefix+".doc").read().strip().split("\n\n")
        title_data_list = open(full_data_file_prefix+".title").read().strip().split("\n\n")
        image_data_list = open(full_data_file_prefix+".image").read().strip().split("\n\n")
        label_data_list = open(full_data_file_prefix+".label.greedyrecall-docfull").read().strip().split("\n\n") # Use collective oracle
        print("Data sizes: %d %d %d %d"%(len(doc_data_list), len(title_data_list), len(image_data_list), len(label_data_list)))
        
        print("Preparing data based on model requirement ...")
        doccount = 0
        for doc_data, title_data, image_data, label_data in zip(doc_data_list, title_data_list, image_data_list, label_data_list):
            
            doc_lines = doc_data.strip().split("\n")
            title_lines = title_data.strip().split("\n")
            image_lines = image_data.strip().split("\n")
            label_lines = label_data.strip().split("\n")
            
            filename = doc_lines[0].strip()

            if ((filename == title_lines[0].strip()) and (filename == image_lines[0].strip()) and (filename == label_lines[0].strip())):
                # Put filename
                self.filenames.append(filename)
                
                # Doc
                thisdoc = []
                for idx in range(FLAGS.max_doc_length):
                    thissent = []
                    if (idx+1) < len(doc_lines):
                        thissent = [int(item) for item in doc_lines[idx+1].strip().split()]
                    thissent = process_to_chop_pad(thissent, FLAGS.max_sent_length)
                    thisdoc.append(thissent)
                self.docs.append(thisdoc)

                # Extract First Sentences form Doc
                thisfirstsentences = []
                for idx in range(FLAGS.max_firstsentences_length):
                    thissent = []
                    if (idx+1) < len(doc_lines):
                        thissent = [int(item) for item in doc_lines[idx+1].strip().split()]
                    thissent = process_to_chop_pad(thissent, FLAGS.max_sent_length)
                    thisfirstsentences.append(thissent)
                self.firstsentences.append(thisfirstsentences)

                # Extract N random Sentences form Doc
                docindices = range(len(doc_lines)-1)
                random.shuffle(docindices)
                thisradomsentences = []
                for idx in range(FLAGS.max_randomsentences_length):
                    thissent = []
                    if idx < len(docindices):
                        thissent = [int(item) for item in doc_lines[docindices[idx]+1].strip().split()]
                    thissent = process_to_chop_pad(thissent, FLAGS.max_sent_length)
                    thisradomsentences.append(thissent)
                self.randomsentences.append(thisradomsentences)

                # Title
                thistitle = []
                for idx in range(FLAGS.max_title_length):
                    thissent = []
                    if (idx+1) < len(title_lines):
                        thissent = [int(item) for item in title_lines[idx+1].strip().split()]
                    thissent = process_to_chop_pad(thissent, FLAGS.max_sent_length)
                    thistitle.append(thissent)
                self.titles.append(thistitle)
    
                # Image
                thisimage = []
                for idx in range(FLAGS.max_image_length):
                    thissent = []
                    if (idx+1) < len(image_lines):
                        thissent = [int(item) for item in image_lines[idx+1].strip().split()]
                    thissent = process_to_chop_pad(thissent, FLAGS.max_sent_length)
                    thisimage.append(thissent)
                self.images.append(thisimage)
                        
                # Labels 1/0, 1, 0 and 2 -> 0 || Weights
                thislabel = []
                thisweight = []
                for idx in range(FLAGS.max_doc_length):
                    thissent_label = 0
                    thissent_weight = 0
                    if (idx+1) < len(label_lines):
                        thissent_label = int(label_lines[idx+1].strip()) 
                        if thissent_label == 2:
                            thissent_label = 0
                        thissent_weight = 1
                    thislabel.append(thissent_label)
                    thisweight.append(thissent_weight)
                self.labels.append(thislabel)
                self.weights.append(thisweight)

            else:
                print("Some problem with %s.* files. Exiting!"%full_data_file_prefix)
                exit(0)
                   
            if doccount%10000==0:
                print("%d ..."%doccount)
            doccount += 1

        # Set Fileindices
        self.fileindices = range(len(self.filenames))

class DataProcessor:
    def prepare_news_data(self, vocab_dict, data_type="training"):
        data = Data(vocab_dict, data_type)
        return data
        
    def prepare_vocab_embeddingdict(self):
        # Numpy dtype
        dtype = np.float16 if FLAGS.use_fp16 else np.float32
        
        vocab_dict = {}
        word_embedding_array = []
        
        # Add padding
        vocab_dict["_PAD"] = PAD_ID
        # Add UNK
        vocab_dict["_UNK"] = UNK_ID
        
        # Read word embedding file
        wordembed_filename = ""
        if FLAGS.anonymized_setting:
            wordembed_filename = FLAGS.pretrained_wordembedding_anonymdata
        else:
            wordembed_filename = FLAGS.pretrained_wordembedding_orgdata
        print("Reading pretrained word embeddings file: %s"%wordembed_filename)

        embed_line = ""
        linecount = 0
        with open(wordembed_filename, "r") as fembedd:
            for line in fembedd:
                if linecount == 0:
                    vocabsize = int(line.split()[0])
                    # Initiate fixed size empty array
                    word_embedding_array = np.empty((vocabsize, FLAGS.wordembed_size), dtype=dtype)    
                else:
                    linedata = line.split()
                    vocab_dict[linedata[0]] = linecount + 1
                    embeddata = [float(item) for item in linedata[1:]][0:FLAGS.wordembed_size]
                    word_embedding_array[linecount-1] = np.array(embeddata, dtype=dtype)
                    
                if linecount%10000 == 0:
                    print(str(linecount)+" ...")
                linecount += 1
        print("Read pretrained embeddings: %s"%str(word_embedding_array.shape))
        
        print("Size of vocab: %d (_PAD:0, _UNK:1)"%len(vocab_dict))
        vocabfilename = ""
        if FLAGS.anonymized_setting:
            vocabfilename = FLAGS.train_dir+"/vocab-anonym"
        else:
            vocabfilename = FLAGS.train_dir+"/vocab-org"
        print("Writing vocab file: %s"%vocabfilename)
        foutput = open(vocabfilename,"w")
        vocab_list = [(vocab_dict[key], key) for key in vocab_dict.keys()]
        vocab_list.sort()
        vocab_list = [item[1] for item in vocab_list]
        foutput.write("\n".join(vocab_list)+"\n")
        foutput.close()
        return vocab_dict, word_embedding_array
    

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
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import math_ops

# from tf.nn import variable_scope
from my_flags import FLAGS
from model_utils import * 

### Various types of extractor

def sentence_extractor_nonseqrnn_noatt(sents_ext, encoder_state):
    """Implements Sentence Extractor: No attention and non-sequential RNN
    Args:
    sents_ext: Embedding of sentences to label for extraction
    encoder_state: encoder_state
    Returns:
    extractor output and logits
    """
    # Define Variables
    weight = variable_on_cpu('weight', [FLAGS.size, FLAGS.target_label_size], tf.random_normal_initializer())
    bias = variable_on_cpu('bias', [FLAGS.target_label_size], tf.random_normal_initializer())
    
    # Get RNN output
    rnn_extractor_output, _ = simple_rnn(sents_ext, initial_state=encoder_state)
    
    with variable_scope.variable_scope("Reshape-Out"):
        rnn_extractor_output =  reshape_list2tensor(rnn_extractor_output, FLAGS.max_doc_length, FLAGS.size)

        # Get Final logits without softmax
        extractor_output_forlogits = tf.reshape(rnn_extractor_output, [-1, FLAGS.size])
        logits = tf.matmul(extractor_output_forlogits, weight) + bias
        # logits: [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
        logits = tf.reshape(logits, [-1, FLAGS.max_doc_length, FLAGS.target_label_size], name='final-logits')
    return rnn_extractor_output, logits

def sentence_extractor_nonseqrnn_titimgatt(sents_ext, encoder_state, titleimages):
    """Implements Sentence Extractor: Non-sequential RNN with attention over title-images
    Args:
    sents_ext: Embedding of sentences to label for extraction
    encoder_state: encoder_state
    titleimages: Embeddings of title and images in the document
    Returns:
    extractor output and logits
    """
    
    # Define Variables
    weight = variable_on_cpu('weight', [FLAGS.size, FLAGS.target_label_size], tf.random_normal_initializer())
    bias = variable_on_cpu('bias', [FLAGS.target_label_size], tf.random_normal_initializer())
  
    # Get RNN output
    rnn_extractor_output, _ = simple_attentional_rnn(sents_ext, titleimages, initial_state=encoder_state)
  
    with variable_scope.variable_scope("Reshape-Out"):
      rnn_extractor_output = reshape_list2tensor(rnn_extractor_output, FLAGS.max_doc_length, FLAGS.size)

      # Get Final logits without softmax
      extractor_output_forlogits = tf.reshape(rnn_extractor_output, [-1, FLAGS.size])
      logits = tf.matmul(extractor_output_forlogits, weight) + bias
      # logits: [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
      logits = tf.reshape(logits, [-1, FLAGS.max_doc_length, FLAGS.target_label_size], name='final-logits')
    return rnn_extractor_output, logits

def sentence_extractor_seqrnn_docatt(sents_ext, encoder_outputs, encoder_state, sents_labels):
    """Implements Sentence Extractor: Sequential RNN with attention over sentences during encoding
    Args:
    sents_ext: Embedding of sentences to label for extraction
    encoder_outputs, encoder_state
    sents_labels: Gold sent labels for training
    Returns:
    extractor output and logits
    """ 
    # Define MLP Variables
    weights = {
      'h1': variable_on_cpu('weight_1', [2*FLAGS.size, FLAGS.size], tf.random_normal_initializer()),
      'h2': variable_on_cpu('weight_2', [FLAGS.size, FLAGS.size], tf.random_normal_initializer()),
      'out': variable_on_cpu('weight_out', [FLAGS.size, FLAGS.target_label_size], tf.random_normal_initializer())
      }
    biases = {
      'b1': variable_on_cpu('bias_1', [FLAGS.size], tf.random_normal_initializer()),
      'b2': variable_on_cpu('bias_2', [FLAGS.size], tf.random_normal_initializer()),
      'out': variable_on_cpu('bias_out', [FLAGS.target_label_size], tf.random_normal_initializer())
      }
    
    # Shift sents_ext for RNN
    with variable_scope.variable_scope("Shift-SentExt"):
        # Create embeddings for special symbol (lets assume all 0) and put in the front by shifting by one
        special_tensor = tf.zeros_like(sents_ext[0]) #  tf.ones_like(sents_ext[0])
        sents_ext_shifted = [special_tensor] + sents_ext[:-1]
        
    # Reshape sents_labels for RNN (Only used for cross entropy training)
    with variable_scope.variable_scope("Reshape-Label"):
        # only used for training
        sents_labels = reshape_tensor2list(sents_labels, FLAGS.max_doc_length, FLAGS.target_label_size)
        
    # Define Sequential Decoder
    extractor_outputs, logits = jporg_attentional_seqrnn_decoder(sents_ext_shifted, encoder_outputs, encoder_state, sents_labels, weights, biases)

    # Final logits without softmax
    with variable_scope.variable_scope("Reshape-Out"):
        logits = reshape_list2tensor(logits, FLAGS.max_doc_length, FLAGS.target_label_size)
        extractor_outputs = reshape_list2tensor(extractor_outputs, FLAGS.max_doc_length, 2*FLAGS.size)

    return extractor_outputs, logits


def policy_network(vocab_embed_variable, document_placeholder, label_placeholder):
    """Build the policy core network.
    Args:
    vocab_embed_variable: [vocab_size, FLAGS.wordembed_size], embeddings without PAD and UNK
    document_placeholder: [None,(FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                 FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.max_sent_length]
    label_placeholder: Gold label [None, FLAGS.max_doc_length, FLAGS.target_label_size], only used during cross entropy training of JP's model.
    Returns:
    Outputs of sentence extractor and logits without softmax
    """
    
    with tf.variable_scope('PolicyNetwork') as scope:
        
        ### Full Word embedding Lookup Variable
        # PADDING embedding non-trainable 
        pad_embed_variable = variable_on_cpu("pad_embed", [1, FLAGS.wordembed_size], tf.constant_initializer(0), trainable=False)
        # UNK embedding trainable
        unk_embed_variable = variable_on_cpu("unk_embed", [1, FLAGS.wordembed_size], tf.constant_initializer(0), trainable=True)     
        # Get fullvocab_embed_variable
        fullvocab_embed_variable = tf.concat(0, [pad_embed_variable, unk_embed_variable, vocab_embed_variable])
        # print(fullvocab_embed_variable)
        
        ### Lookup layer
        with tf.variable_scope('Lookup') as scope:
            document_placeholder_flat = tf.reshape(document_placeholder, [-1])
            document_word_embedding = tf.nn.embedding_lookup(fullvocab_embed_variable, document_placeholder_flat, name="Lookup")
            document_word_embedding = tf.reshape(document_word_embedding, [-1, (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                                                                FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), 
                                                                           FLAGS.max_sent_length, FLAGS.wordembed_size])
            # print(document_word_embedding) 
          
        ### Convolution Layer
        with tf.variable_scope('ConvLayer') as scope:
            document_word_embedding = tf.reshape(document_word_embedding, [-1, FLAGS.max_sent_length, FLAGS.wordembed_size])
            document_sent_embedding = conv1d_layer_sentence_representation(document_word_embedding) # [None, sentembed_size]
            document_sent_embedding = tf.reshape(document_sent_embedding, [-1, (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                                                                FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.sentembed_size])
            # print(document_sent_embedding)
            
        ### Reshape Tensor to List [-1, (max_doc_length+max_title_length+max_image_length), sentembed_size] -> List of [-1, sentembed_size]
        with variable_scope.variable_scope("ReshapeDoc_TensorToList"):
            document_sent_embedding = reshape_tensor2list(document_sent_embedding, (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                                                                    FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.sentembed_size)  
            # print(document_sent_embedding) 
            
        # document_sents_enc 
        document_sents_enc = document_sent_embedding[:FLAGS.max_doc_length]
        if FLAGS.doc_encoder_reverse:
            document_sents_enc = document_sents_enc[::-1]
            
        # document_sents_ext
        document_sents_ext = document_sent_embedding[:FLAGS.max_doc_length]

        # document_sents_titimg
        document_sents_titimg = document_sent_embedding[FLAGS.max_doc_length:]

        ### Document Encoder 
        with tf.variable_scope('DocEnc') as scope:
            encoder_outputs, encoder_state = simple_rnn(document_sents_enc)
      
        ### Sentence Label Extractor
        with tf.variable_scope('SentExt') as scope:
            if (FLAGS.attend_encoder) and (len(document_sents_titimg) != 0):
                # Multiple decoder
                print("Multiple decoder is not implement yet.")
                exit(0)
                # # Decoder to attend captions
                # attendtitimg_extractor_output, _ = simple_attentional_rnn(document_sents_ext, document_sents_titimg, initial_state=encoder_state)                
                # # Attend previous decoder
                # logits = sentence_extractor_seqrnn_docatt(document_sents_ext, attendtitimg_extractor_output, encoder_state, label_placeholder)
                
            elif (not FLAGS.attend_encoder) and (len(document_sents_titimg) != 0):
                # Attend only titimages during decoding
                extractor_output, logits = sentence_extractor_nonseqrnn_titimgatt(document_sents_ext, encoder_state, document_sents_titimg)

            elif (FLAGS.attend_encoder) and (len(document_sents_titimg) == 0):
                # JP model: attend encoder
                extractor_outputs, logits = sentence_extractor_seqrnn_docatt(document_sents_ext, encoder_outputs, encoder_state, label_placeholder)
                
            else:
                # Attend nothing
                extractor_output, logits = sentence_extractor_nonseqrnn_noatt(document_sents_ext, encoder_state)

    # print(extractor_output)
    # print(logits)
    return extractor_output, logits   

def cross_entropy_loss(logits, labels, weights):
    """Estimate cost of predictions
    Add summary for "cost" and "cost/avg".
    Args:
    logits: Logits from inference(). [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
    Returns:
    Cross-entropy Cost
    """
    with tf.variable_scope('CrossEntropyLoss') as scope:
        # Reshape logits and labels to match the requirement of softmax_cross_entropy_with_logits
        logits = tf.reshape(logits, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
        labels = tf.reshape(labels, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels) # [FLAGS.batch_size*FLAGS.max_doc_length]
        cross_entropy = tf.reshape(cross_entropy, [-1, FLAGS.max_doc_length])  # [FLAGS.batch_size, FLAGS.max_doc_length]
        if FLAGS.weighted_loss:
            cross_entropy = tf.mul(cross_entropy, weights)
      
        # Cross entroy / document
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) # [FLAGS.batch_size]
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='crossentropy')

        # ## Cross entroy / sentence
        # cross_entropy_sum = tf.reduce_sum(cross_entropy)
        # valid_sentences = tf.reduce_sum(weights)
        # cross_entropy_mean = cross_entropy_sum / valid_sentences
        
        # cross_entropy = -tf.reduce_sum(labels * tf.log(logits), reduction_indices=1)
        # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='crossentropy')
    
        tf.add_to_collection('cross_entropy_loss', cross_entropy_mean)
        # # # The total loss is defined as the cross entropy loss plus all of
        # # # the weight decay terms (L2 loss).
        # # return tf.add_n(tf.get_collection('losses'), name='total_loss')     
    return cross_entropy_mean

### Training functions

def train_cross_entropy_loss(cross_entropy_loss):
    """ Training with Gold Label: Pretraining network to start with a better policy
    Args: cross_entropy_loss
    """
    with tf.variable_scope('TrainCrossEntropyLoss') as scope:
        
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='adam') 
        
        # Compute gradients of policy network
        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="PolicyNetwork")
        # print(policy_network_variables)
        grads_and_vars = optimizer.compute_gradients(cross_entropy_loss, var_list=policy_network_variables)
        # print(grads_and_vars)

        # Apply Gradients
        return optimizer.apply_gradients(grads_and_vars)

### Accuracy Calculations

def accuracy(logits, labels, weights):
  """Estimate accuracy of predictions
  Args:
    logits: Logits from inference(). [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
  Returns:
    Accuracy: Estimates average of accuracy for each sentence
  """
  with tf.variable_scope('Accuracy') as scope:
    logits = tf.reshape(logits, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
    labels = tf.reshape(labels, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
    correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1)) # [FLAGS.batch_size*FLAGS.max_doc_length]
    correct_pred =  tf.reshape(correct_pred, [-1, FLAGS.max_doc_length])  # [FLAGS.batch_size, FLAGS.max_doc_length]
    correct_pred = tf.cast(correct_pred, tf.float32)
    # Get Accuracy
    accuracy = tf.reduce_mean(correct_pred, name='accuracy')
    if FLAGS.weighted_loss:
      correct_pred = tf.mul(correct_pred, weights)      
      correct_pred = tf.reduce_sum(correct_pred, reduction_indices=1) # [FLAGS.batch_size]
      doc_lengths = tf.reduce_sum(weights, reduction_indices=1) # [FLAGS.batch_size]
      correct_pred_avg = tf.div(correct_pred, doc_lengths)
      accuracy = tf.reduce_mean(correct_pred_avg, name='accuracy')
  return accuracy


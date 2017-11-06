# source ~/.bash_rc
# source ~/.bash_profile
# activate-tensorflow
# export TMP=/tmp/shashi-tmp-clulow-gpu2

# python document_summarizer_gpu2.py --tmp_directory /tmp/shashi-tmp-clulow-gpu2 --max_image_length 10 --train_dir /disk/ocean/snarayan/Document-Summarization/Experiment-With-RL/cnn-simplecrossentropy-collectiveoracle-att-img > /disk/ocean/snarayan/Document-Summarization/Experiment-With-RL/cnn-simplecrossentropy-collectiveoracle-att-img/train.log

# python document_summarizer_gpu2.py --tmp_directory /tmp/shashi-tmp-clulow-gpu2 --max_title_length 1 --max_image_length 10 --train_dir /disk/ocean/snarayan/Document-Summarization/Experiment-With-RL/cnn-simplecrossentropy-collectiveoracle-att-tit-img > /disk/ocean/snarayan/Document-Summarization/Experiment-With-RL/cnn-simplecrossentropy-collectiveoracle-att-tit-img/train.log

python document_summarizer_gpu2.py --tmp_directory /tmp/shashi-tmp-clulow-gpu2 --max_title_length 1 --max_image_length 10 --train_dir /disk/ocean/snarayan/Document-Summarization/Experiment-With-RL/cnn-simplecrossentropy-collectiveoracle-att-tit-img --model_to_load 8 --exp_mode test > /disk/ocean/snarayan/Document-Summarization/Experiment-With-RL/cnn-simplecrossentropy-collectiveoracle-att-tit-img/test.log

# python document_summarizer_gpu2.py --tmp_directory /tmp/shashi-tmp-clulow-gpu2 --max_image_length 10 --max_firstsentences_length 1 --train_dir /disk/ocean/snarayan/Document-Summarization/Experiment-With-RL/cnn-simplecrossentropy-collectiveoracle-att-img-fs > /disk/ocean/snarayan/Document-Summarization/Experiment-With-RL/cnn-simplecrossentropy-collectiveoracle-att-img-fs/train.log


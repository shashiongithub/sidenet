## Neural Extractive Summarization with Side Information

This repository releases codes for SideNet (Neural Extractive
Summarization with Side Information). They use Tensorflow 0.10, please
use scripts provided by Tensorflow to translate them to newer
upgrades.

Please contact me at shashi.narayan@ed.ac.uk for any question.

Please cite this paper if you use any of these:

**Neural Extractive Summarization with Side Information, Shashi
Narayan, Nikos Papasarantopoulos, Shay B. Cohen, Mirella Lapata, ILCC,
School of Informatics, University of Edinburgh, arXiv:1704.04530
(preprint)**

> Most extractive summarization methods focus on the main body of the
> document from which sentences need to be extracted.  The gist of the
> document often lies in the side information of the document, such as
> title and image captions. These types of side information are often
> available for newswire articles. We propose to explore side
> information in the context of single document extractive
> summarization. We develop a framework for single-document
> summarization composed of a hierarchical document encoder and an
> attentionbased extractor with attention over side information.  We
> evaluate our models on a large scale news dataset. We show that
> extractive summarization with side information consistently
> outperforms its counterpart (that does not use any side information),
> in terms on both informativeness and fluency.


### The CNN and DM  dataset (Hermann et al 2015) with Side Information ###

Dataset with sideinfo: http://kinloch.inf.ed.ac.uk/public/cnn-dm-sideinfo-data.zip

Dataset with oracle labels: http://kinloch.inf.ed.ac.uk/public/cnn-dm-sidenet-oracle.zip

### Demonstration ###

Live Demo: http://kinloch.inf.ed.ac.uk/sidenet.html


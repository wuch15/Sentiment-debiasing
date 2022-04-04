# Sentiment-debiasing
Source codes for sentiment-debiasing in news recommendation.

1. Baic Environment Requirements
* Ubuntu 16.04
* Anaconda with Python 3.6.9

2. Additional Packages
* tensorflow==1.15.0
* Keras==2.2.4

2. Hardware requirements
Needs a server with at least one GPU with a memory of at least 8GB.

3. Training and Testing
* Download the MIND dataset from the website https://msnews.github.io/ and the GloVe embedding from https://nlp.stanford.edu/projects/glove/.
* Change the path names and data file names. 
* Run the cells in "run.ipynb"

Note: The logs at the training stage will show the training loss and accuracy. Logs at the test stage will show the test results in terms of ranking performance and sentiment orientation. The training process usually takes 1~2 hrs on a GPU machine.



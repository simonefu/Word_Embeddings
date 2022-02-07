# Word_Embeddings

It is an example of how to compute word embeddings from a scratch implementation or using the pre build library (Gensim)
It contains a library to compute GLove and Word2vec (Cbow and Skipgram) from scratch.
The main are in the jupyter notebook. 

## Installation

In order to use the libraries (library_name = 'glove_utils', 'word2vec_utils') for the custom notebook you need to build the library as follows:

from the /main 
```bash
cd lib/{library_name}
python3 setup.py bdist_wheel
pip3 install dist/{library_name}-0.1.0-py3-none-any.whl (--force-reinstall)
```
## Usage

```python
import {library_name}
```
and follows the examples in the jupyter notebook

## Requiriments

The requiriments to run the example are specified in the /main/requiriments.txt

## Dataset

Some text examples are shown in the dataset dir. It also includes an example of the wikipedia italian corpus.

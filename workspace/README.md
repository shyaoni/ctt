# Retrival-Based Chatbot with Topic Transition 

## Requirements

+ Python 3
+ pytorch 0.4

## Instruction

### Package Installation
    
    There is no extra package required if you use Anaconda environment.

    Run the following command to initialize nltk downloads.

### Data Preparation

#### ConvAI2 dataset

+ Install [ParlAI](https://github.com/facebookresearch/ParlAI)
+ Download dataset following the [instructions](https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2).
+ Copy the data from `<ParlAI\_dir>/data` to `data/convai2/source/`.

You can also download the data from our [google drive distribution](https://drive.google.com/file/d/1HvlB0wz5DvZru5qcm3r2J1Fu9F-JxwGJ/view?usp=sharing).

## Usage

To train a model, run the command

    python train.py --config config.retrival_baseline --dts dts_ConvAI2 --cuda

For testing or evaluating, change `train.py` to `test.py` or `eval.py` in the
above command.

Here:

+ `--dts` specifies the dataset.
+ `--config` speicifies the module path to the configuration. Not note to include the `.py` suffix. Four configs are provided.
    + `config.retrival_baseline` uses bi-directional RNNs as the encoders.
    + `config.retrival_attention_isinput` add attention layer basing on the baseline. 

## Results

The table shows results of recall@1 

| retrival\_baseline | retriva\_attention |
| -------------------| -------------------|
| 0.286              | \                  |

## Notification

### Cache

Data and initial word embedding will be cached once they are prepared. Remove the cache file to re-build specified one.  

### Custom configuration

You can slightly change the experiment setting by modifying the configuration files.

#### Word Embedding 

Word embedding should be provided following [GloVE](https://nlp.stanford.edu/projects/glove/)'s format. In our settings, Twitter 2B with 200d vectors is used. 

#### Stopwords 

Stopwords should be provided following format in [igorbrigadir/stopwords](https://github.com/igorbrigadir/stopwords). In our settings, [SMART](https://github.com/igorbrigadir/stopwords/blob/master/en/smart.txt) is used.

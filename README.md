# Curriculum-Meta Learning for Order-robust Continual Relation Extraction
This repo is an implemetation of the paper [**Curriculum-Meta Learning for Continual Relation Extraction**](https://arxiv.org/abs/2101.01926)
## Pretrain files
Because the size of pretrained embedding file is too large to upload, you can download and extract **glove.6B.300d.txt** file from the [Link](http://nlp.stanford.edu/data/glove.6B.zip).

## Requirements
You can use following script to install the dependencies.

    pip install -r requirements.txt

## Usage
To run the model on the FewRel dataset with default arguments, you can simply run the `run.py` file as following.

    python run_fewrel.py

Change arguments in the `__main__` method of `run.py` can lead to different experiment settings.

Following script can run the experiment on the FewRel dataset with setting, which uses cuda device 1, random seed 100, random task spliting, 100 instances for relation, batch size 50 and sequence offset 3.

```
python run.py --cuda_id=1 --random_seed=100 --task_arrange=random --train_instance_num=100 -batch_size=50 --sequence_index=3
```

## Acknowledgement
This repo is published for the double-blind review.

## Code and data for the paper "Learning from Both Structural and Textual Knowledge for Inductive Knowledge Graph Completion"

### Prerequisites

 * Python 3.8
 * pytorch==1.10.0
 * TensorFlow==1.15.0 (for LSTK-NeuralLP and LSTK-DRUM)


### Datasets
We use seven datasets in our experiments.

| Datasets           | Download Links (original)         |
|--------------------|-----------------------------------|
| HacRED          | https://github.com/qiaojiim/HacRED   |
| DocRED          | https://github.com/thunlp/DocRED     |
| BioRel          | https://bit.ly/biorel_dataset        |

### Models
We use four models in our experiments.

| Models             | Code Download Links (original)                  |
|--------------------|-------------------------------------------------|
| NeuralLP           | https://github.com/fanyangxyz/Neural-LP         |
| DRUM               | https://github.com/alisadeghian/DRUM            |
| RNNLogic           | https://github.com/DeepGraphLearning/RNNLogic   |
| TELM               | This work                                       |

## Use examples

### The first stage
LSTK is a two-stage framework. In the first stage, it aims at generating a set of soft triples for reasoning.

You can generate a set of soft triples by:

Path for code: ``src_nli``

1. training a textual entailment model:
```
$ python main_nli.py [dataset]
```

2. Searching triples with corresponding texts:
```
$ python generate_triples_by_index.py [dataset]
```
3. If the dataset is in Chinese, use:
```
$ python generate_triples_by_index_zh.py [dataset]
```
4. Appling the trained textual entailment model to generate soft triples:
```
$ python apply_model_nli.py [dataset]
```

After the above process, you can get three files (train/valid/test_triple_scores.txt) storing soft triples.

You can also directly download our processed soft triples:

| Datasets           | Download Links (processed)         |
|--------------------|------------------------------------|
| HacRED             | [Google Drive](https://drive.google.com/file/d/1qpxhtaHXYqvE5mtpTG2cUEImMUI_YSrW/view?usp=sharing)    |
| DocRED             | [Google Drive](https://drive.google.com/file/d/1rR_Ryr5qydT34XtF1plRRa6c4j5VjCD5/view?usp=sharing)    |
| BioRel             | [Google Drive](https://drive.google.com/file/d/1zpL9Q4gZ96OqQTduoHoS2Q-fCgxi6Xln/view?usp=sharing)    |

### The second stage
In the second stage, you can use the generated soft triples to train SOTA neural approximate rule-based models.

#### LSTK-TELM

Path for code: ``src/LSTK-TELM``

The script for both training and evaluation on the HacRED dataset is:
```
bash run_hacred.sh
```

The script for both training and evaluation on the HacRED dataset is:
```
bash run_docred.sh
```

The script for both training and evaluation on the BioRel dataset is:
```
bash run_biorel.sh
```

The script for rule extraction is:
```
bash run_rules.sh [dataset]
```

We also provide the runing scripts of baseline methods:


#### LSTK-NeuralLP and LSTK-DRUM

Path for code: ``src/LSTK-NeuralLP`` or ``src/LSTK-DRUM``

The training script is:
```
python python -u src/main.py --datadir=[dataset]/ --exp_name=[dataset] --num_step 4 --gpu 0 --exps_dir exps --max_epoch 100 --seed 1234
```

The evaluation script is:
```
sh eval/collect_all_facts.sh [dataset]

python eval/get_truths.py [dataset]

python eval/evaluate.py --preds=exps/[dataset]/test_predictions.txt --truths=[dataset]/truths.pckl
```

#### LSTK-RNNLogic

Path for code: ``src/LSTK-RNNLogic``

The script for environment installation is:
```
cd LSTK-RNNLogic/codes/pyrnnlogiclib/
python setup.py install
```

The script for data preparation is:
```
python process_dicts.py
python get_scores.py
python process_soft.py
```

The script for both training and evaluation is:
```
python run.py --data_path [dataset] --num_generated_rules 2000 --num_rules_for_test 500 --num_important_rules 0 --prior_weight 0.01 --cuda --predictor_learning_rate 0.1 --generator_epochs 5000 --max_rule_length 2
```


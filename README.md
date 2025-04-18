pip install torch torchvision
pip install numpy matplotlib
pip install sentencepiece
pip install nltk

- python BPETokenizer.py
runs and generates the tokenizer model and vocab. To train again will need to delete the olde model and vocab files. Expects data to be on data/raw just like the template project structure.

- python train.py 
Trains all three models. The parameters can the set inside the files. Automatically saves the best checkpoint for each model. Will skip training if model already exists, so need to delete the model to retrain. Will generate plots in a new folder called plots.

- python evaluation.py
Will generate the Perplexity and BLEU scores. And then generate the prompt response. The prompts can be changed inside the file.

- data files: Expects to be in the same structure as the template project.
	- data/raw/\*.txt
	- data/train.jsonl
	- data/test.jsonl


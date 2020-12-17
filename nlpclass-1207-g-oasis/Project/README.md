**Final Project: Text Summarization from basic to advanced approaches**

**Group name**: Oasis

**Group members**: Siyu Wu (301395909) ; Nattapat Juthaprachakul (301350117)

**Email**: swa246@sfu.ca ; njuthapr@sfu.ca

**Assignments of works**:

1. Processing dataset ： Nattapat
2. Implementing vanilla LSTM sequence-to-sequence model： Nattapat
3. Implementing models with transformers: Nattapat, Siyu
4. Evaluation: Nattapat, Siyu
5. Making video, PPT and final report: Nattapat, Siyu


**Text Summarization from basic to advanced approaches**

- The goal of this project is to implement different Text summarization techniques on the same dataset and compare them based on the same evaluation metrics such as ROUGE.

- The program is written in Python.
- This project is implemented in Google Colab using CPU, GPU, and TPU.
- Data preprocessing is implemented in process_data.py file and the results are saved into pickle.
- Predicted summaries and evaluation results are saved into CSV files for each model.

**Video Presentation:**

https://vault.sfu.ca/index.php/s/3CbZSqq0abUshGv


**Requirements:**

1. Tensorflow 2.0, Keras (LSTM model)
2. Huggingface Transformers API (Transformer models)
3. Pytorch (Finetuned Transformer model)
4. Python ML Libraries eg. Pandas, Numpy
5. Python 3.6 and above


**In this project, we are using the following models:**

1. Vanilla LSTM sequence-to-sequence model (trained from scratch)
2. BART-base model (finetuned)
3. BART-large model
4. T5-small model
5. T5-base model
6. T5-large model

- The code is in project.ipynb (or in separated files.)

**Datasets:**

- CNN and Daily Mail datasets (one online source: https://cs.nyu.edu/~kcho/DMQA/)

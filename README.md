# CEMIL: Classifier Expansion with Multi-vIew LLM Knowledge

This is the code repository for Classifier Expansion with Multi-vIew LLM Knowledge (CEMIL).

## Preparation

1. Download the extracted fine-tuned CUB, AWA2, and SUN features.

2. Download the fine-tuned ResNet101 features for CUB, AWA2, and SUN datasets.
The image datasets are available at the following links: [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/), [AWA2](https://cvml.ista.ac.at/AwA2/), and [SUN](https://groups.csail.mit.edu/vision/SUN/hierarchy.html). However, this paper does not utilize the original images.

3. Modify the default values for "--dataroot" and "--rootpath" in `main.py` to specify the path to your data and the directory for saving outputs, respectively.

4. Prepare the API key for LLMs, and modify it in `LLM.py`, or use the pre-queried results in the `LLM` folder.

## Usage

### Step 1: LLM-based Descriptive Text Generation

We start by generating descriptive texts and embeddings for the datasets based on LLMs.

After modify the API key in `LLM.py` to your OpenAI API key or other LLM, run the following script to generate descriptive texts and embeddings for the datasets:

```bash
$ python gen_desc.py
```

In default, the script will generate descriptive texts and embeddings for the datasets CUB, AWA2, and SUN. The LLM could be chosen from gpt4o, gpt4omini, llama, and qwen_plus. The embedding model could be chosen from clip, sbert, qwen, and llama. You can modify the dataset list in the script.

The script will save the generated texts in the `LLM` folder, and the embeddings in the `embeddings` folder.

### Step 2: Feature Refining and Cross-modal Learning

In this step, we refine the features and perform cross-modal feature learning to expand the classifier.

To complete the learning process for the classifier expansion task, just select the appropriate parameters and run the `main.py` script.

The code will automatically prepare the required data, initiate the learning process, and perform a comprehensive evaluation on both standard and generalized zero-shot learning.
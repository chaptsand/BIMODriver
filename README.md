# Integrating Large Model-Generated Biological Knowledge and Multi-Omics Features for Cancer Driver Gene Identification
BIMODriver is a novel cancer driver gene identification method that leverages biological knowledge-guided prompts for large language models (LLMs) and integrates multiomics data via contrastive learning.

This repo is for the source code of "Integrating Large Model-Generated Biological Knowledge and Multi-Omics Features for Cancer Driver Gene Identification". \
Paper Link: 

## Instructions

This project contains all the codes for BIMODriver algorithms to experiment on the CPDB and String databases, respectively.

Setup
------------------------
The setup process for BIMODriver requires the following steps:
### Download
Download Bioprompt.  The following command clones the current Bioprompt repository from GitHub:

    git clone https://github.com/weiba/BIMODriver.git

### Environment Settings
> python=3.9.19
>
> torch==2.0.1+cu118
>
> numpy==1.26.4
>
> pandas==2.2.1
>
> ollama==0.3.3
>
> scipy==1.13.0
>
> scikit-learn==1.4.2

Ollama list

> NAME            ID              SIZE
>
> gemma2:latest   ff02c3702f32    5.4 GB
>

GPU: GeForce RTX 3090 24G	CUDA Version: 12.0

CPU: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz

### Usage

#### Model composition and meaning

BIMODriver is composed of LLM modules and experimental modules.

The ”LLM“ folder contains detailed records of the data preprocessing and the part where the output of the large language model is called.

The "BIMODriver" folder contains the running code related to our model. Among them, model.py contains the model settings, and main.py is the running code.

#### Step 1: Process Data

Run the GO data processing notebook to generate foundational GO data sentences:

> jupyter notebook LLM/go_data_process.ipynb

Run the "LLM-go-part.py" to generate three functional dimensions - Biological Process (BP), Molecular Function (MF), and Cellular Component (CC)：

`python LLM/LLM-go-part.py`

Run the "LLM-satment.py" to generate the three sentences for the summary of the large language model - self, neighbor, together

`python LLM/LLM-go-part.py`

Run the "LLM/embedding.ipynb" notebook to generate the embedding

> jupyter notebook LLM/embedding.ipynb



#### Step 2: Run BIMODriver Framework

Run the BIMODriver Framework using the following command line:

> python BIMODriver/main.py
>
> 

## Results

The framework has been validated on pan-cancer and 15 individual cancer datasets, demonstrating improved performance over existing models. Ablation studies confirm the critical role of GO-guided prompts in generating valuable gene semantic information.

## Project Structure

> BIMODriver /
>
> ├─BIMODriver				# Main algorithm framework
>│  └─__pycache__
> ├─data					   #The multi-omics data used in the method and the constructed semantic network
>│  ├─CPDB
> │  │  └─Specific cancer
>│  ├─cpdb_network_LLM
> │  ├─STRING
>│  └─string_network_LLM	
> └─LLM						# Large Language Model components



# Contact

```
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com). 
```




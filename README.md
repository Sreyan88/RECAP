# RECAP: Retrieval-Augmented Audio Captioning

This is the official repository for the paper **RECAP: Retrieval-Augmented Audio Captioning** accepted at ICASSP 2024 for oral presentation.

[[`Paper`](https://ieeexplore.ieee.org/abstract/document/10448030)] [[`CLAP Checkpoints`](https://drive.google.com/drive/folders/1gcvboyMj_p0jj1CNhR2ZzuJYZ7Qrcw8v?usp=sharing)] [[`Weakly labeled captions for AudioSet, AudioCaps, and Clotho`](https://drive.google.com/drive/folders/1RL5RJ6FP3UbFYXh0N848As4Qc4KvDHXA?usp=sharing)]  

**Note**: CLAP checkpoints are meant to be used with the models and code defined in [this repository](https://github.com/LAION-AI/CLAP/tree/main).

We present RECAP (REtrieval-Augmented Audio CAPtioning), a novel and effective audio captioning system that generates captions conditioned on an input audio and other captions similar to the audio retrieved from a datastore. Additionally, our proposed method can transfer to any domain without the need for any additional fine-tuning. To generate a caption for an audio sample, we leverage an audio-text model CLAP to retrieve captions similar to it from a replaceable datastore, which are then used to construct a prompt. Next, we feed this prompt to a GPT-2 decoder and introduce cross-attention layers between the CLAP encoder and GPT-2 to condition the audio for caption generation. Experiments on two benchmark datasets, Clotho and AudioCaps, show that RECAP achieves competitive performance in in-domain settings and significant improvements in out-of-domain settings. Additionally, due to its capability to exploit a large text-captions-only datastore in a _training-free_ fashion, RECAP shows unique capabilities of captioning novel audio events never seen during training and compositional audios with multiple events. To promote research in this space, we also release 150,000+ new weakly labeled captions for AudioSet, AudioCaps, and Clotho.
![image](https://github.com/Sreyan88/RECAP/blob/main/assets/RECAP_2-1.png)

## Setup
1. You are required to install the dependencies: `pip install -r requirements.txt`. If you have [conda](https://www.anaconda.com) installed, you can run the following: 

```shell
cd RECAP && \
conda create -n recap python=3.10 && \
conda activate recap && \
pip install -r requirements.txt
```

2. After updating the paths in [recap.sh](https://github.com/Sreyan88/RECAP/blob/main/recap.sh), run the following command:
```shell
bash recap.sh
```

The repository has both training and inference commands. We recommend doing them one by one. Once you run `python train.py` and save a checkpoint, update the paths in `python infer.py` and infer your trained model. `--model_path` refers to the parent folder where your checkpoints are saved, and `--checkpoint_path` refers to the checkpoint that you want to use (the training code saves multiple checkpoints, one every time a predefined number of steps is completed).  

## Using CLAP Checkpoints
Once you have downloaded our [CLAP checkpoints](https://drive.google.com/drive/folders/1gcvboyMj_p0jj1CNhR2ZzuJYZ7Qrcw8v?usp=sharing), you can use them for evaluation using [CLAP](https://github.com/LAION-AI/CLAP/tree/6b1b4b5b4b87f4e19d3836d2ae7d7272e1c69410/src/laion_clap/evaluate).

## Acknowledgements
Our codebase has been inspired by [SmallCap](https://github.com/RitaRamo/smallcap). We thank the authors for open-sourcing [their work](https://openaccess.thecvf.com/content/CVPR2023/papers/Ramos_SmallCap_Lightweight_Image_Captioning_Prompted_With_Retrieval_Augmentation_CVPR_2023_paper.pdf).

## License


## Citation
```BibTeX
@INPROCEEDINGS{10448030,
  author={Ghosh, Sreyan and Kumar, Sonal and Reddy Evuru, Chandra Kiran and Duraiswami, Ramani and Manocha, Dinesh},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Recap: Retrieval-Augmented Audio Captioning}, 
  year={2024},
  volume={},
  number={},
  pages={1161-1165},
  keywords={Training;Signal processing;Benchmark testing;Acoustics;Decoding;Feeds;Speech processing;Automated audio captioning;multimodal learning;retrieval-augmented generation},
  doi={10.1109/ICASSP48485.2024.10448030}}
```

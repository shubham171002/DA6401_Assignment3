# Sequence-to-Sequence Transliteration using RNNs and Attention (Dakshina Dataset - Gujarati)

**Submitted By** : Shubham Shah

**Roll No.** : DA24M020

This repository contains the complete implementation of a character-level sequence-to-sequence transliteration system using RNNs (Vanilla RNN, LSTM, GRU) with and without attention mechanisms. The model has been trained and evaluated on the Gujarati subset of the Dakshina dataset, tackling the task of converting Latin-script inputs to native Gujarati script.

Following problems have been covered in this assignment work : 
1. Model building & complexity analysis (Q1)
2. Hyperparameter tuning with Weights & Biases sweeps (Q2)
3. Test‑set evaluation & sample predictions (Q3)
4. Attention‑augmented decoder and heatmap visualizations (Q5)
5. Connectivity visualization between encoder and decoder states (Q6)

## Project Structure
```
├── Vanilla/
│   ├── data.py              # Data loading, tokenization, DataLoader
│   ├── vocab.py             # CharVocab: encode/decode utilities
│   ├── models.py            # Encoder, Decoder, Seq2Seq (RNN/GRU/LSTM)
│   ├── train.py             # train_loop, eval_loop functions
│   ├── test_eval.py         # Exact‑match test evaluation script
│   └── predictions_vanilla/
│       ├── vanilla_best.pth     # Best checkpoint from sweep
│       └── vanilla_prediction.tsv  # test‑set predictions
│  
├── Attention/
│   ├── data.py              # (same as Vanilla)
│   ├── vocab.py             # (same as Vanilla)
│   ├── models.py            # + BahdanauAttention, AttentionDecoder
│   ├── train.py             # train_attention, evaluate_attention
│   ├── test_eval.py         # evaluate vanilla vs. attention
│   ├── sample_grid.py       # Generates and saves 3×3 attention heatmaps
│   ├── Question_6.py        # Connectivity visualization script
│   └── predictions_attention/
│       ├── best_attn_model.pth    # Best attention model checkpoint
│       └── attention_heatmaps.png # Sample attention heatmaps
│
└── sweep_config.py         # Updated hyperparameter sweep configuration
```

## Prerequisites
Language : Python 3.8+

Install dependencies via:
```
pip install torch torchvision wandb matplotlib scikit-learn
```

## Experimental Setup
* **Language**: Gujarati (gu) from Dakshina dataset
* **Task**: Latin to native-script transliteration
* **Models**: Vanilla RNN, GRU, LSTM; Attention-enhanced LSTM
* **Loss**: CrossEntropyLoss (ignoring <pad>)
* **Optimization**: Adam, gradient clipping
* **Evaluation**: Exact-match accuracy at sequence level

## Data Preparation 
1. Download the Dakshina dataset and place the Gujarati folder here:
```
repo-root/
└── data/
    └── dakshina_dataset_v1.0/
        └── gu/lexicons/{train.tsv,dev.tsv,test.tsv}
```

2. All scripts assume the path ```data/dakshina_dataset_v1.0/gu/lexicons/```

## How to Run ?
**1. Vanilla Model**
* Hyperparameter Tuning (Q2)
```
wandb sweep sweep_config.py --project DA6400_A3  # for non attention 
wandb agent <SWEEP_ID>
```

* Test set Evaluation (Q3)
```
cd Vanilla
python test_eval.py --checkpoint predictions_vanilla/vanilla_best.pth --output predictions_vanilla/vanilla_prediction.tsv
```

* Inspect Prediction
Compare ```vanilla_prediction.tsv``` against ```test.tsv``` for exact‑match and error analysis.

**2. Attention Model**
* 1. Train with Attention (Q5)
```
cd Attention
python train.py
```

* Generate Attention Heatmap
```
python sample_grid.py --checkpoint predictions_attention/best_attn_model.pth --output attention_heatmaps.png
```

* Connectivity Visualization (Q6)
```
python Question_6.py --checkpoint predictions_attention/best_attn_model.pth --output connectivity.png
```

## Experiment Tracking with Wandb : 
https://wandb.ai/da24m020-iit-madras/DA6401_A3/reports/DA6401-Assignment-3-Report---VmlldzoxMjgxNTM2Mw?accessToken=tj85y0mtm4wn2c8h7aa3g09xjvtinspyz7bhvp86nlhxjms9qy043h4hcfkodxsy

## Results & Analysis 
* Q1: Computed total Flops and parameter counts for RNN, LSTM, and GRU seq2seq.
* Q2–3: Identified best hyperparameters via sweep (emb_size=128, hidden_size=64, etc.), logged train/val loss & accuracy. Achieved ~38% validaition and test acuracy (non attention)
* Q4: Achieved ~41% validation accuracy; test accuracy validated with exact‑match script (with attention mechanism).
* Q5: Attention improved error correction; visualized with heatmaps.
* Q6: Connectivity plots reveal which encoder positions most influence each decoder step.

## Extending & Reproducing 
*  To rerun the sweep with updated ranges, edit ```sweep_config.py``` and relaunch.
*  To try a different Dakshina language, point ```--lang <xx>``` or modify ```get_dataloaders('<lang>')```.
*  All experiments are GPU‑ready; for AMP and multi‑worker loading, refer to comments in ```train.py```.

## Learnings & Takeaways
* Vanilla Seq2Seq models struggle on long sequences without attention
* Exact-match accuracy is highly sensitive to decoding (greedy vs beam)
* WandB sweeps helped find optimal configurations with minimal manual tuning
* Heatmaps and vector similarity visualizations are powerful tools to understand model behavior

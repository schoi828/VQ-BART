# Fine-tuning BART with vector quantization inspired from [VQVAE](https://github.com/MishaLaskin/vqvae))

This repository is built on top of [Fairseq](https://github.com/facebookresearch/fairseq)

### 1) Preprocess the data using BPE by following instructions [here](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md)

### 2) To fine tune VQ layer with pretrained BART, run train.sh

```bash
EMBDIM=32             #Codebook vector dimension
MLPS=2                #number of MLP layers before and after VQ layer
BSZ=32                #batch size
EPOCH=30              #numbrt of epochs to train
CODEBOOK=$((512*4))   #number of codebook vectors
GPUS=4                #number of GPUS to use
ARCHTYPE=vqbart       #vqbart for default vq model. gaubart for sqvae variant
CRIT=vq_cross_entropy 
NAME="$EMBDIM"_"$MLPS"_"$CODEBOOK"_"$ARCHTYPE"
ARCH="$ARCHTYPE"_"large"

CUDA_VISIBLE_DEVICES="0,1,2,3" python -O train.py ./data-bin/wikitext-103 \
--mask 0.3 \
--tokens-per-sample 512 \
--total-num-update 500000 \
--max-update 500000 \
--warmup-updates 10000 \
--task denoising \
--save-interval 1 \
--optimizer adam \
--lr-scheduler polynomial_decay \
--lr 0.0004 \
--dropout 0.1 \
--max-tokens 3200 \
--weight-decay 0.01 \
--attention-dropout 0.1 \
--share-all-embeddings \
--clip-norm 0.1 \
--skip-invalid-size-inputs-valid-test \
--log-format json \
--log-interval 50 \
--save-interval-updates 500 \
--keep-interval-updates 1 \
--update-freq 4 \
--seed 4 \
--distributed-world-size $GPUS \
--distributed-port 54187 \
--mask-length span-poisson \
--replace-length 1 \
--encoder-learned-pos \
--decoder-learned-pos \
--rotate 0.0 \
--mask-random 0.1 \
--permute-sentences 1 \
--insert 0.0 \
--poisson-lambda 3.5 \
--dataset-impl mmap \
--bpe gpt2 \
--num-workers 4 \
--distributed-init-method tcp://localhost:54187 \
--log-file logs_$NAME.txt \
--arch $ARCH \
--criterion $CRIT \
--codebook $CODEBOOK \
--max-epoch $EPOCH \
--emb-dim $EMBDIM \
--MLPLayers $MLPS \
--batch-size $BSZ \
--tensorboard-logdir logs_$NAME \
--save-dir checkpoints/$NAME \
--disable-validation \
```

### 3) To run GLUE task with trained model:

Follow the instructions [here](https://github.com/facebookresearch/fairseq/edit/main/examples/bart/README.glue.md
) to preprocess for GLUE tasks.

Then, run GLUE.sh with hyperparameters in the table [here](https://github.com/facebookresearch/fairseq/edit/main/examples/bart/README.glue.md
).

### 4) After finetuning a model, to train a codebook sampling transformer run train_transformer.sh:

```bash
EMBDIM=32             #Codebook vector dimension
MLPS=2                #number of MLP layers before and after VQ layer
BSZ=1                 #batch size
EPOCH=10              #numbrt of epochs to train
CODEBOOK=$((512*4))   #number of codebook vectors
TEMBD=$((256*3))      #transformer embedding vector dimension
MODEL="$EMBDIM"_"$MLPS"_"$CODEBOOK"
NAME="$MODEL"_uncond
LR=4.5e-06           #learning rate
GPUS=4               #number of GPUS to use

CUDA_VISIBLE_DEVICES="0,1,2,3" python -O train.py ./data-bin/wikitext-103 \
--tokens-per-sample 200 \
--total-num-update 500000 \
--max-update 500000 \
--warmup-updates 10000 \
--task denoising \
--save-interval 1 \
--optimizer adam \
--lr-scheduler polynomial_decay \
--lr $LR \
--dropout 0.1 \
--max-tokens 200 \
--weight-decay 0.01 \
--attention-dropout 0.1 \
--share-all-embeddings \
--clip-norm 0.1 \
--skip-invalid-size-inputs-valid-test \
--log-format json \
--log-interval 50 \
--save-interval-updates 500 \
--keep-interval-updates 1 \
--update-freq 4 \
--seed 4 \
--distributed-world-size $GPUS \
--distributed-port 54187 \
--mask-length span-poisson \
--replace-length 1 \
--encoder-learned-pos \
--decoder-learned-pos \
--rotate 0.0 \
--insert 0.0 \
--dataset-impl mmap \
--bpe gpt2 \
--num-workers 4 \
--distributed-init-method tcp://localhost:54187 \
--log-file logs_$NAME.txt \
--arch uncond_transformer \
--criterion uncond_cross_entropy \
--codebook $CODEBOOK \
--max-epoch $EPOCH \
--emb-dim $EMBDIM \
--MLPLayers $MLPS \
--batch-size $BSZ \
--tensorboard-logdir logs_$NAME \
--save-dir checkpoints/$NAME \
--vocab-size $CODEBOOK \
--n-embd $TEMBD \
--disable-validation
```

### 5) Interpolation with trained models:

To run interpolation on trained modelS, run the following command:
```bash
python sample_test.py --checkpoint_path $PATH-TO-CHECKPOINT --checkpoint_file $CHECKPOINT-FILE --beam $BEAM-SEARCH-SIZE
```

```python
text1 = 'I walked in the past.'#"While the planets move in elliptical orbits around the Sun, their orbits are not very elliptical"
text2 = 'The organ of Corti is well protected from accidental injury'#"<mask> a passage of Lorem Ipsum, you need to be sure <mask> anything <mask> hidden in the middle of text."
text3 = 'Our computer can carry us in time as well as in space'
text_inp = [text2,text3]
output = model.fill_mask(text_inp, topk=args.topk, beam=args.beam, match_source_len=args.match_source_len, interpolate=[0.5,0.5])

print(checkpoint_path,output)
```

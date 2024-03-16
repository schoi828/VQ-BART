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


#to resume training --restore-file $CHECKPOINT_PATH

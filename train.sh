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

#to resume training --restore-file $CHECKPOINT_PATH

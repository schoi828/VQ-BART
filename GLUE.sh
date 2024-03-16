TOTAL_NUM_UPDATES=40000  # total-num-update
WARMUP_UPDATES=1858      # 6 percent of the number of updates
LR=5e-06                 # Peak LR for polynomial LR scheduler.
NUM_CLASSES=3
MAX_SENTENCES=128        # Batch size.
BART_PATH=checkpoints/16_2_1024/checkpoint10.pt
CODEBOOK=$((1024))
EMBDIM=16
MLPs=2
TASK=QNLI
NAME=$EMBDIM-$MLPs-$CODEBOOK-$TASK

CUDA_VISIBLE_DEVICES="1,2,3" python train.py $TASK-bin/ \
    --restore-file $BART_PATH \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --add-prev-output-tokens \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --arch vqbart_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --threshold-loss-scale 1 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy \
    --codebook $CODEBOOK --emb-dim $EMBDIM --MLPLayers $MLPs \
    --log-file $NAME-logs.txt \
    --tensorboard-logdir logs/$NAME \
    --save-dir checkpoints/$NAME \
    --maximize-best-checkpoint-metric;

#--fp16 \
#--restore-file checkpoints/8_2_e2e_re/checkpoint_6_16500.pt \
#--no-epoch-checkpoints \
#--e2e \
#--max_valid_steps 100\
#    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \

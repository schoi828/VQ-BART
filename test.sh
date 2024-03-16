CUDA_VISIBLE_DEVICES=3, python -O test.py ./data-bin/wikitext-103 \
--mask 0.3 \
--tokens-per-sample 512 \
--task denoising \
--dropout 0.1 \
--max-tokens 3200 \
--attention-dropout 0.1 \
--share-all-embeddings \
--seed 4 \
--replace-length 1 \
--encoder-learned-pos \
--decoder-learned-pos \
--rotate 0.0 \
--insert 0.0 \
--poisson-lambda 3.5 \
--bpe gpt2 \
--arch vqbart_large \
--criterion vq_cross_entropy \
--codebook $((512*4)) \
--emb-dim 8 \
--MLPLayers 2 \
--restore-file checkpoints/8_2_2048/checkpoint_2_5500.pt \

#--fp16 \
#--restore-file checkpoints/8_2_e2e_re/checkpoint_6_16500.pt \
#--no-epoch-checkpoints \
#--e2e \
#--max_valid_steps 100\
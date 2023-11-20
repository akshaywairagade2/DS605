CUDA_VISIBLE_DEVICES=0,  python  -m torch.launch --nproc_per_node=1  --master_port=12346  \
	train.py C:\Users\akshay\Downloads\SSF-main\SSF-main\cifar-100-python --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
    --batch-size 16 --epochs 50 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-5 --warmup-epochs 2  \
    --lr 1e-3 --min-lr 1e-5 \
    --drop-path 0 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output  output/vit_base_patch16_224_in21k/cifar_100/ssf \
	--amp --tuning-mode ssf --pretrained  \

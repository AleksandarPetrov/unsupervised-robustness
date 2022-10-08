
set -x

echo "Training finetune_untargeted_upgd"
python3 main_moco.py \
  -a vit_small -b 128 \
  --optimizer=adamw --lr=1.5e-3 --weight-decay=.1 \
  --start-epoch 300 --epochs=310 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 32 --resume ../data/models/vit-s-300ep.pth.tar \
  --checkpoint-dir "../data/models/moco3_finetune_untargeted_upgd" \
  --attack pgd-untargeted --ball-size 0.05 --attack-iterations 3 --attack-alpha 0.005 \
  ../data/imagenet

echo "\n\n\nTraining finetune_targeted_upgd"
python3 main_moco.py \
  -a vit_small -b 128 \
  --optimizer=adamw --lr=1.5e-3 --weight-decay=.1 \
  --start-epoch 300 --epochs=310 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 32 --resume ../data/models/vit-s-300ep.pth.tar \
  --checkpoint-dir "../data/models/moco3_finetune_targeted_upgd" \
  --attack pgd-targeted --ball-size 0.05 --attack-iterations 3 --attack-alpha 0.005 \
  ../data/imagenet

echo "\n\n\nTraining finetune_loss_upgd"
python3 main_moco.py \
  -a vit_small -b 128 \
  --optimizer=adamw --lr=1.5e-3 --weight-decay=.1 \
  --start-epoch 300 --epochs=310 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --workers 32 --resume ../data/models/vit-s-300ep.pth.tar \
  --checkpoint-dir "../data/models/moco3_finetune_loss_upgd" \
  --attack l-pgd --ball-size 0.05 --attack-iterations 3 --attack-alpha 0.005 \
  ../data/imagenet

set +x

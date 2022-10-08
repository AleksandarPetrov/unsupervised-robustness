#!/bin/bash
echo "What would you like to run"

# Operating system names are used here as a data source
select command in finetune_untargeted_upgd finetune_targeted_upgd finetune_loss_upgd 
do
case $command in


"finetune_untargeted_upgd")
set -x
python3 main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 192 --moco-k 65280 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --epochs 210 --workers 32 --resume ../data/models/moco_v2_200ep_pretrain.pth.tar \
  --checkpoint-dir "../data/models/finetune_untargeted_upgd" \
  --attack pgd-untargeted --ball-size 0.05 --attack-iterations 3 --attack-alpha 0.005 \
  ../data/imagenet
set +x
break
;;

"finetune_targeted_upgd")
set -x
python3 main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 192 --moco-k 65280 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --epochs 210 --workers 32 --resume ../data/models/moco_v2_200ep_pretrain.pth.tar \
  --checkpoint-dir "../data/models/finetune_targeted_upgd" \
  --attack pgd-targeted --ball-size 0.05 --attack-iterations 3 --attack-alpha 0.005 \
  ../data/imagenet
set +x
break
;;

"finetune_loss_upgd")
set -x
python3 main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 192 --moco-k 65280 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --epochs 210 --workers 32 --resume ../data/models/moco_v2_200ep_pretrain.pth.tar \
  --checkpoint-dir "../data/models/finetune_loss_upgd" \
  --attack l-pgd --ball-size 0.05 --attack-iterations 3 --attack-alpha 0.005 \
  ../data/imagenet
set +x
break
;;

# Matching with invalid data
*)
echo "Invalid entry."
break
;;
esac
done

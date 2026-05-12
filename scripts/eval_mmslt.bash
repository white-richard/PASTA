# TODO: removed logic in pasta by mistake 
.venv/bin/python src/train_pasta.py \
    --eval \
    --resume best_checkpoint_phoenix.pth \
    --language_decoder mbart \
    --vision_backbone resnet18 \
    --config src/configs/config_mmslt_phoenix.yaml \
    --output_dir out/mmslt_eval \
    --eval-metrics \
    --num_workers 4 \
    --eval_num_workers 2 \
    --batch-size 16

# python -m visdom.server
# !./scripts/run_Recycle_gan_cpu.sh
python train.py --dataroot ./flowers/01/ --name test --niter 1 --niter_decay 0 --display_freq 10 --print_freq 10 --save_latest_freq 10 --model recycle_gan --which_model_netG resnet_6blocks --which_model_netP unet_256 --dataset_mode unaligned_triplet --no_dropout --gpu 0 --gpu_ids -1 --identity 0 --pool_size 0 --nThreads 0
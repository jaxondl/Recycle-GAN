# python -m visdom.server
# sh ./scripts/run_Recycle_gan_test_cpu.sh
python test.py --dataroot ./flowers/01/ --name test --model cycle_gan --which_model_netG resnet_6blocks --dataset_mode unaligned --no_dropout --gpu 0 --gpu_ids -1 --how_many 200 --loadSize 256 --nThreads 0
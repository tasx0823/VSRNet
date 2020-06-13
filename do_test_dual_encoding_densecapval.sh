rootpath=/ssd1/vis/sunx/densecap
testCollection=densecapval
logger_name=ckpt/log26
n_caption=5
overwrite=0

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tester.py $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --n_caption $n_caption


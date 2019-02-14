import argparse
import os
import train 
import tensorflow as tf 
#os.environ['CUDA_VISIBLE_DEVICES']='0'
#tf.device('/gpu:0')

def main():
    parser = argparse.ArgumentParser(description="manual to this script")
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--epoch',type=int,default=20)
    parser.add_argument('--class_n',type=int,default=10)
    parser.add_argument('--dataset',default="MNIST")
    

    args = parser.parse_args()

    if args.dataset=="MNIST":
        source="MNIST"
        target="USPS"
        source_dir="./Log/ADDA/source_network/best/MNIST/NOBN"
        adv_dir="./Log/ADDA/advermodel/best/MNIST2USPS/NOBN"
    elif args.dataset=="vaihinghen":
        source="area3"
        target="area23"
        source_dir="./Log/ADDA/source_network/best/area3"
        adv_dir="./Log/ADDA/advermodel/best/area3toarea23"


    if args.step == 1:
        train.step1(source=source,epoch=args.epoch,
            classes_num=args.class_n,logdir=source_dir)
        return 
    elif args.step == 2:
        train.step2(source=source,target=target,epoch=args.epoch,
            classes_num=args.class_n,source_dir=source_dir,
            logdir=adv_dir)
        return 
    elif args.step == 3:
        train.step3(source=source,target=target,logdir=adv_dir,
            classes_num=args.class_n)
        return 

if __name__ == "__main__":
    main()
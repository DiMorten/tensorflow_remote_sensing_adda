import argparse
import os
import train 
import tensorflow as tf 
os.environ['CUDA_VISIBLE_DEVICES']='1'
tf.device('/gpu:1')

def main():
    parser = argparse.ArgumentParser(description="manual to this script")
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--epoch',type=int,default=20)
    parser.add_argument('--class_n',type=int,default=10)
    parser.add_argument('--dataset',default="MNIST")
    

    args = parser.parse_args()

    if args.dataset=="MNIST":
        source="MNIST"
    elif args.dataset=="vaihinghen":
        source="area3"



    if args.step == 1:
        train.step1(source=source,epoch=args.epoch,
            classes_num=args.class_n)
        return 
    elif args.step == 2:
        train.step2(source="MNIST",target="USPS",epoch=args.epoch)
        return 
    elif args.step == 3:
        train.step3("MNIST","USPS")
        return 

if __name__ == "__main__":
    main()
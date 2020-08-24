import argparse
import os
from model import attentiongan
import tensorflow as tf
#tf.set_random_seed(20)


def get_params():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='ckpt', help='checkpoint name')
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
    parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam') # default=0.0002
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.0, help='beta1 momentum term of adam')
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.99, help='beta2 momentum term of adam')
    parser.add_argument('--phase', dest='phase', default='train', help='train, test') # 
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
    parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
    parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')

    args = parser.parse_args()
    return args


def main():
    args = get_params()
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        args = get_params()
        model = attentiongan(sess, args)
        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'test':
            model.test(args)

if __name__ == '__main__':
    main()

import os
import argparse
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from model import *
from utils import *

def test(epoch, dataloader, net, m, alpha):
    accum_loss = 0
    net.eval()
    with torch.no_grad():
      for img, cls in dataloader:
          img, cls = [x.cuda() for x in (img, cls)]

          b = net(img)
          loss = hashing_loss(b, cls, m, alpha)
          accum_loss += float(loss)

    accum_loss /= len(dataloader)
    print(f'[{epoch}] val loss: {accum_loss:.4f}')
    return accum_loss


def main():
    parser = argparse.ArgumentParser(description='train DSH')
    parser.add_argument('--imagenet', default=None, help='Use ImageNet instead of CIFAR')

    parser.add_argument('--weights', default='', help="path to weight (to continue training)")

    parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
    parser.add_argument('--ngpu', type=int, default=0, help='which GPU to use')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for the image loader')

    parser.add_argument('--binary_bits', type=int, default=14, help='length of hashing binary')
    parser.add_argument('--alpha', type=float, default=0.01, help='weighting of regularizer')
    parser.add_argument('output', help='Output .csv file')

    opt = parser.parse_args()
    print(opt)

    choose_gpu(opt.ngpu)
    assert( opt.imagenet is not None)
    assert( opt.weights )

    train_loader, test_loader = init_imagenet_dataloader(opt.imagenet, opt.batchSize,opt.workers,indexing=True)
    logger = SummaryWriter()

    # setup net
    net = DSH(opt.binary_bits)
    print(net)

    print(f'loading weight form {opt.weights}')
    net.load_state_dict(torch.load(opt.weights, map_location=lambda storage, location: storage))
    net.cuda()

    bs=[]
    clses=[]
    dummy=torch.pow(2,torch.arange(opt.binary_bits)).unsqueeze(0).cuda()
    with torch.no_grad():
        for img, c in train_loader:
            # convert into a binary code
            out=torch.sign(net(img.cuda())).long().clamp(0,1)
            bs.extend( [ int(i) for i in  torch.sum(out*dummy,1).cpu() ] )
            clses.extend( [ int(i) for i in c] )

    # save index
    pfx=opt.imagenet+os.sep
    with open(opt.output,'w') as f:
        f.write("idx,cls,hash\n")

        for b,c,i in zip(bs, clses, train_loader.dataset.samples ):
            f.write("{},{},{}\n".format(i[0].replace(pfx,'') ,i[1], b))

if __name__ == '__main__':
    main()

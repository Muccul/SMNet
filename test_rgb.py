import cv2
import os
import argparse
import glob
from torch.autograd import Variable
from utils import *
from model.net import SMNet

# python test_rgb.py -c 40 --mode S --test_noiseL 15 --test_data McMaster
parser = argparse.ArgumentParser(description="SMNet_rgb Test")
parser.add_argument("--checkpoint", "-c", type=int, default="40", help='checkpoint of model')
parser.add_argument("--test_data", type=str, default='McMaster', choices=["McMaster", "CBSD68", "Kodak24"], help='test on McMaster,CBSD68 or Kodak24')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
parser.add_argument("--mode", type=str, default="S", choices=['S', 'B'], help='with known noise level (S) or blind training (B)')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = SMNet(in_channels=3)
    model = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    if opt.mode == 'S':
        model.load_state_dict(torch.load(os.path.join("weights","model_rgb_L%d_%d.pth" %(opt.test_noiseL, opt.checkpoint))))
    else:
        model.load_state_dict(torch.load(os.path.join("weights", "model_rgb_B_%d.pth" % (opt.checkpoint))))
    model.eval()

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*'))
    files_source.sort()

    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img = torch.tensor(Img)

        Img = Img.permute(2,0,1)
        Img = Img.numpy()
        Img = np.tile(Img,(3,1,1,1))  #expand the dimensional
        Img = np.float32(normalize(Img))
        ISource = torch.Tensor(Img)

        # noise
        torch.manual_seed(12) 
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        ISource = Variable(ISource)
        INoisy = Variable(INoisy)
        ISource= ISource.cuda() 
        INoisy = INoisy.cuda() 
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()

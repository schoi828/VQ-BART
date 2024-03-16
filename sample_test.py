import sys
import os

current_path = sys.path[0]
sys.path.insert(0,os.path.join(current_path,'fairseq/model'))
import fairseq.models as models
from fairseq.models.bart import BARTModel, BARTHubInterface, VQLayer
from fairseq.tasks.denoising import DenoisingTask
import torch
import numpy as np
from fairseq import tasks, options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from omegaconf import OmegaConf
from fairseq.data import (
    Dictionary,
    LanguagePairDataset,
    TransformEosDataset,
    data_utils,
    noising,
)
import argparse


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='train args')
  parser.add_argument('--seed', type=int, default=0,
                      help='random seed')
  parser.add_argument('--match_source_len',action='store_true', help='output texts match source texts length')
  parser.add_argument('--cuda', type=str, default='0',
                      help='gpu id')
  parser.add_argument('--vanilla', action='store_true',
                      help='run original bart-large model')
  parser.add_argument('--checkpoint_path',type=str,default='checkpoints/4_2_512')
  parser.add_argument('--checkpoint_file',type=str,default='checkpoint10.pt')
  parser.add_argument('--beam', type=int, default=3)
  parser.add_argument('--topk', type=int, default=3)
      
  args = parser.parse_args()

#fix random seed
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

#set device
device = torch.device('cuda:'+args.cuda)
sys.path[0] = current_path

#set checkpoints
checkpoint_path = args.checkpoint_path
checkpoint_file = args.checkpoint_file

#load models to run interpolation
model = torch.hub.load('pytorch/fairseq', 'bart.large') if args.vanilla \
    else BARTModel.from_pretrained(checkpoint_path,checkpoint_file,vq=True)
model = model.to(device)

text1 = 'I walked in the past.'#"While the planets move in elliptical orbits around the Sun, their orbits are not very elliptical"
text2 = 'The organ of Corti is well protected from accidental injury'#"<mask> a passage of Lorem Ipsum, you need to be sure <mask> anything <mask> hidden in the middle of text."
text3 = 'Our computer can carry us in time as well as in space'
text_inp = [text2,text3]
output = model.fill_mask(text_inp, topk=args.topk, beam=args.beam, match_source_len=args.match_source_len, interpolate=[0.5,0.5])

print(checkpoint_path,output)

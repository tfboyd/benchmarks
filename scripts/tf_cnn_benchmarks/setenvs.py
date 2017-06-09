from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
class arglist :
    cpu = 'bdw'
    model = 'alexnet'
    data_dir = None
def setenvs(inpargv):
    args = arglist()
    for i in range(0,len(inpargv)-1) :
        if inpargv[i] == '--cpu' :
            args.cpu = inpargv[i+1]
        elif inpargv[i] == '--model' :            
            args.model = inpargv[i+1]
        elif inpargv[i] == '--data_dir' :            
            args.data_dir = inpargv[i+1]    
    assert (args.cpu == 'knl' or args.cpu == 'bdw')
    if (args.cpu == 'bdw' and args.model == 'alexnet') :
        os.environ["KMP_BLOCKTIME"] = "30"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    elif (args.cpu == 'knl' and args.model == 'alexnet'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "34"
        else:
          os.environ["OMP_NUM_THREADS"]= "136"         
    elif (args.cpu == 'bdw' and args.model == 'googlenet'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]="granularity=fine,verbose,compact,1,0"
    elif (args.cpu == 'knl' and args.model == 'googlenet'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "34"
        else:
          os.environ["OMP_NUM_THREADS"]= "68"
    elif (args.cpu =='bdw' and args.model == 'vgg11'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    elif (args.cpu =='knl' and args.model == 'vgg11'):    
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "34"
        else:
          os.environ["OMP_NUM_THREADS"]= "68"
    elif (args.cpu =='bdw' and args.model == 'inception3'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]="granularity=fine,verbose,compact,1,0"     
    elif (args.cpu =='knl' and args.model == 'inception3'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "50"
        else:
          os.environ["OMP_NUM_THREADS"]= "68"       
    elif (args.cpu =='bdw' and args.model == 'resnet50'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]="granularity=fine,verbose,compact,1,0"
    elif (args.cpu =='knl' and args.model == 'resnet50'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "50"
        else:
          os.environ["OMP_NUM_THREADS"]= "136"    
    return args

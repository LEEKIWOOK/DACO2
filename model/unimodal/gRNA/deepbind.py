from scipy.stats import bernoulli
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      
# input of shape(batch_size,inp_chan,iW)
class ConvNet(nn.Module):
    def __init__(self, param, glen): 
        super(ConvNet, self).__init__()

        self.poolType = param["pool"]
        self.neuType = param["neuType"]
        self.dropprob = param["dropprob"]

        self.hnode = param["hnode"]
        self.sigmaConv = param["sigmaConv"]
        self.sigmaNeu = param["sigmaNeu"]

        #self.batch_size = batch_size
        self.glen = glen
        
        self.wConv=torch.randn(self.hnode, 4 , self.glen).to(device)
        torch.nn.init.normal_(self.wConv,mean=0,std=self.sigmaConv)
        self.wConv.requires_grad=True
        
        self.wRect=torch.randn(self.hnode).to(device)
        torch.nn.init.normal_(self.wRect)
        self.wRect=-self.wRect
        self.wRect.requires_grad=True

        self.wHidden=torch.randn(2 * self.hnode, 2 * self.hnode).to(device)
        self.wHiddenBias=torch.randn(2 * self.hnode).to(device)
        
        if self.neuType=='nohidden':
            
            if self.poolType=='maxavg':
                self.wNeu=torch.randn(2*self.hnode,1).to(device)
            else:
                self.wNeu=torch.randn(self.hnode,1).to(device)

            self.wNeuBias=torch.randn(1).to(device)
            torch.nn.init.normal_(self.wNeu,mean=0,std=self.sigmaNeu)
            torch.nn.init.normal_(self.wNeuBias,mean=0,std=self.sigmaNeu)

        else:
            if self.poolType=='maxavg':
                self.wHidden=torch.randn(2 * self.hnode, 2 * self.hnode).to(device)
            else:
                self.wHidden=torch.randn(self.hnode, 2 * self.hnode).to(device)
            
            self.wNeu=torch.randn(2 * self.hnode,1).to(device)
            self.wNeuBias=torch.randn(1).to(device)
            self.wHiddenBias=torch.randn(2 * self.hnode).to(device)

            torch.nn.init.normal_(self.wNeu,mean=0,std=self.sigmaNeu)
            torch.nn.init.normal_(self.wNeuBias,mean=0,std=self.sigmaNeu)
            torch.nn.init.normal_(self.wHidden,mean=0,std=0.3)
            torch.nn.init.normal_(self.wHiddenBias,mean=0,std=0.3)
            
            self.wHidden.requires_grad=True
            self.wHiddenBias.requires_grad=True

        self.wNeu.requires_grad=True
        self.wNeuBias.requires_grad=True
    
   
    # def divide_two_tensors(self,x):
    #     l=torch.unbind(x)

    #     list1=[l[2*i] for i in range(int(x.shape[0]/2))]
    #     list2=[l[2*i+1] for i in range(int(x.shape[0]/2))]
    #     x1=torch.stack(list1,0)
    #     x2=torch.stack(list2,0)
    #     return x1,x2
    def forward_pass(self,x,mask=None,use_mask=False):
        
        conv=F.conv1d(x, self.wConv, bias=self.wRect, stride=1, padding=0)
        rect=conv.clamp(min=0)
        maxPool, _ = torch.max(rect, dim=2)
        
        if self.poolType=='maxavg':
            avgPool= torch.mean(rect, dim=2)                          
            pool=torch.cat((maxPool, avgPool), 1)
        else:
            pool=maxPool

        if(self.neuType=='nohidden'):
            if self.mode=='training': 
                if  not use_mask:
                  mask=bernoulli.rvs(self.dropprob, size=len(pool[0]))
                  mask=torch.from_numpy(mask).float().to(device)
                pooldrop=pool*mask
                out=pooldrop @ self.wNeu
                out.add_(self.wNeuBias)
            else:
                out=self.dropprob*(pool @ self.wNeu)
                out.add_(self.wNeuBias)       
        else:
            hid=pool @ self.wHidden
            hid.add_(self.wHiddenBias)
            hid=hid.clamp(min=0)

            if self.mode=='training': 
                if  not use_mask:
                  mask=bernoulli.rvs(self.dropprob, size=len(hid[0]))
                  mask=torch.from_numpy(mask).float().to(device)
                hiddrop=hid*mask
                out=self.dropprob*(hid @ self.wNeu)
                out.add_(self.wNeuBias)
            else:
                out=self.dropprob*(hid @ self.wNeu)
                out.add_(self.wNeuBias) 
        return out,mask
       
    def forward(self, x):
        
        out,_ = self.forward_pass(x)
            
        # else:
        #     x1,x2=self.divide_two_tensors(x)
        #     out1,mask=self.forward_pass(x1)
        #     out2,_=self.forward_pass(x2,mask,True)
        #     out=torch.max(out1, out2)
     
        return out
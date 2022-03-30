import torch
import pickle as pkl
from torchvision import transforms
from torch.utils.data import DataLoader
from core import *

class Actor():
    
    @staticmethod
    def load_pytorch_model(cuda, file):
        if cuda:
            out = torch.load(file)
        else:
            out = torch.load(file, map_location="cpu")
        return(out)
    
    @staticmethod
    def spawn_PAM(file):
        encoder = GuideHRNN(21, 32, 3360, n_classes=5, use_cuda=torch.cuda.is_available()) if "LSTM" in file else GuideHN(21, 32, 1360, n_classes=5, use_cuda=torch.cuda.is_available())
        model = DKL(encoder, [1,5*32])
        return(model)
        
    @staticmethod
    def spawn_offtarget(file):
        encoder = GuideHN2d(
            23, capsule_dimension=32, n_routes=1600, n_classes=5, n_channels=2, use_cuda=torch.cuda.is_available()
        )
        model = DKL(encoder, [1,5*32])
        return(model)
        
    @staticmethod
    def load_pytorch_model(cuda, file):
        model = None
        if "geCRISPR" not in file:
            if "OT" not in file:
                model = Actor.spawn_PAM(file)
            else:
                model = Actor.spawn_offtarget(file)
        else:
            model = Actor.spawn_no_PAM(file)
        if cuda:
            model = model.cuda()
            weights = torch.load(file)
        else:
            weights = torch.load(file, map_location="cpu")
        model.load_state_dict(weights)
        model = model.eval()
        model.return_encoded = True
        return(model)
    
    def __init__(
        self, capsnets, pca, pam, use_pam, before, 
        guide_length, cuda, cut_at_start=0, cut_at_end=0, on=True
    ):
        self.capsnets = []
        self.fullpath = '/home/kwlee/Projects_gflas/DACO2/tools/uace/src/'
        #self.offtarget = Actor.load_pytorch_model(cuda, offtarget)
        for a in capsnets:
            print("model location : " + self.fullpath+a)
            self.capsnets.append(
                Actor.load_pytorch_model(cuda, self.fullpath+a),
            )
        self.onTargetP = on
        if self.onTargetP:
            with open(self.fullpath+pca, "rb") as ih:
                self.pca = pkl.load(ih)
        self.cuda = cuda
        self.pam = pam
        self.use_pam = use_pam
        self.before = before
        self.cut_at_start = cut_at_start
        self.cut_at_end = cut_at_end
        self.guide_length = guide_length
        if on:
            self.transformer = transforms.Compose(
                [
                    OneHotAndCut(
                        self.pam.replace("[ATGC]", "N"), self.use_pam, 
                        self.before, cut_at_start=cut_at_start,
                        cut_at_end=cut_at_end
                    ), 
                    ToTensor(cuda)
                ]
            )
        else:
            self.transformer = transforms.Compose(
                [
                    ImperfectMatchTransform(
                        self.pam.replace("[ATGC]", "N"), self.use_pam, 
                        self.before
                    ), 
                    ToTensor(cuda)
                ]
            )
        
    @staticmethod
    def out_sketches():
        out2 = {
            "guides": [],
            "#offtargets": [],
            "low_activity_ots": [],
            "low_variance_ots": [],
            "low_activity_low_variance_ots": [],
        }
        out1 = {
            "guides": [],
            "activities": [],
            "strands": [],
            "PCA1": [],#PCA first coordinate
            "PCA2": [],#PCA second coordinate
            "starts": [],#starts in gene
            "ends": [],#ends in gene,
            "guides_and_pams": [],
            "PAMs": [],
            "variances": [],
        }
        return(out1, out2)
    
    def compute_ontarget(self, transformed):
		#print(transformed.shape)
        if len(self.capsnets) == 2:
            internal, reconstruction, encoded = self.capsnets[0].forward(
                transformed#.reshape(1, *ss)
            )
            likelihood1 = self.capsnets[0].likelihood(
                internal
            )
            prediction1 = likelihood1.mean.mean(0).cpu().data.numpy()
            variance1 = likelihood1.variance.mean(0).cpu().data.numpy()
            internal, reconstruction, encoded = self.capsnets[1].forward(
                transformed#.reshape(1, *ss)
            )
            likelihood2 = self.capsnets[1].likelihood(
                internal
            )
            prediction2 = likelihood2.mean.mean(0).cpu().data.numpy()
            variance2 = likelihood2.variance.mean(0).cpu().data.numpy()
            prediction = 0.5*(prediction1+prediction2)
            variance = variance1+variance2
        else:
            internal, reconstruction, encoded = self.capsnets[0].forward(
                transformed#.reshape(1, *ss)
            )
            likelihood = self.capsnets[0].likelihood(
                internal
            )
            prediction = likelihood.mean.mean(0).cpu().data.numpy()
            variance = likelihood.variance.mean(0).cpu().data.numpy()
        return(prediction, variance, encoded)
    
    def process_ontarget_fasta(self, fdl):
        out = Actor.out_sketches()[0]
        for guide, start, end, pam, guidepam, strand, transformed in fdl:  #error
            #ss = transformed.shape
            prediction, variance, encoded = self.compute_ontarget(transformed)
            ss = encoded.shape
            pca = self.pca.transform(
                encoded.cpu().data.numpy().reshape(ss[0], ss[1]*ss[2])
            )
            out["PCA1"].extend([float(a) for a in pca.T[0]])
            out["PCA2"].extend([float(a) for a in pca.T[1]])
            out["guides"].extend(guide)
            out["strands"].extend([int(a) for a in strand.cpu().data.numpy()])
            out["starts"].extend([int(a) for a in start.cpu().data.numpy()])
            out["ends"].extend([int(a) for a in end.cpu().data.numpy()])
            out["PAMs"].extend(pam)
            out["guides_and_pams"].extend(guidepam)
            out["activities"].extend([float(a) for a in prediction])
            out["variances"].extend([float(a) for a in variance])
        return(out)
    
    def on(self, fasta, batch_size=256):
        #

        fs = FastaSet(fasta, self.pam, self.guide_length, self.before, self.transformer, self.use_pam)
        fdl = DataLoader(fs, shuffle=False, batch_size=batch_size)
        out = self.process_ontarget_fasta(fdl)
        return(out)
    
    def off_cof(self, cof, batch_size=256, actcut=0.15, varcut=0.015):
        fdl = DataLoader(cof, shuffle=False, batch_size=batch_size)
        out = Actor.out_sketches()[1]
        Y_hat, Y_var = cof.compute_results(self.capsnets[0], fdl)
        o = pd.DataFrame({"y_hat": Y_hat, "y_var": Y_var})
        df = pd.merge(cof.S, o, left_index=True, right_index=True)
        props = cof.compute_proportions(
            df, actcut, varcut
        )
        for a in props:
            out["guides"].append(a)
            out["low_activity_ots"].append(props[a][0])
            out["low_variance_ots"].append(props[a][1])
            out["low_activity_low_variance_ots"].append(props[a][2])
            out["#offtargets"].append(props[a][3])
        return(out)

    def off_ff(self, ffs, batch_size=256, actcut=0.15, varcut=0.015):
        fdl = DataLoader(ffs, shuffle=False, batch_size=batch_size)
        out = Actor.out_sketches()[1]
        for a,b in zip(ffs.S["target"].values, ffs.S["otCount"].values):
            out["guides"].append(a)
            out["#offtargets"].append(int(b))
        out["activities"] = []
        activities = []
        out["variances"] = []
        variances = []
        for transformed in fdl:
            ss = transformed.shape
            prediction, variance, _ = self.compute_ontarget(transformed)
            activities.extend(prediction)
            variances.extend(variance)
        out["activities"] = [float(a) for a in activities]
        out["variances"] = [float(a) for a in variances]
        goods = [0]*len(out["guides"])
        safes = [0]*len(out["guides"])
        sags = [0]*len(out["guides"])
        Y_hat, Y_var = ffs.compute_results(self.capsnets[0], fdl)
        props = ffs.compute_proportions(
            Y_hat, Y_var, ffs.numbers, actcut, varcut
        )
        for a in ffs.no_offtargets:
            props[a] = [1, 1, 1]
        for a in props:
            goods[a] = props[a][0]
            safes[a] = props[a][1]
            sags[a] = props[a][2]
        out["low_activity_ots"] = goods
        out["low_variance_ots"] = safes
        out["low_activity_low_variance_ots"] = sags
        return(out)

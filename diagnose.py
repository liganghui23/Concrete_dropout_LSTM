# -*- coding: utf-8 -*-

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model import Model

def acc_(target_mean, mean):
    pred=mean.argmax(1)
    tru=target_mean.argmax(1)
    num_correct = torch.eq(pred, tru).sum()

    return torch.true_divide(num_correct,mean.shape[0])

def acc_loss(target_mean, mean):
    cost=-torch.sum(target_mean*torch.log(mean),axis=1)

    return torch.mean(cost)

def KL_ID( target_mean, mean, precision,alpha_target=200, smoothing=0.001,epsilon=1e-8):
    target_precision = alpha_target * torch.ones([mean.shape[0], 1]).to(device)
    target_mean=target_mean*(1-target_mean.shape[1]*smoothing)
    target_mean+=smoothing*torch.ones_like(target_mean)

    cost1 = torch.lgamma(target_precision + epsilon)-torch.lgamma(precision + epsilon)
    cost2= torch.sum(torch.lgamma(mean * precision + epsilon) - torch.lgamma(target_mean * target_precision + epsilon), axis=1)
    cost3= torch.sum((target_precision * target_mean - mean * precision) * (torch.digamma(target_mean * target_precision + epsilon) 
                                                                       -torch.digamma(target_precision + epsilon)), axis=1)

    cost = torch.mean(cost1+cost2+cost3)
    return cost

def KL_OD( mean, precision,alpha_target=10,epsilon=1e-8):
    target_precision =alpha_target* torch.ones([mean.shape[0], 1]).to(device)
    
    target_mean=torch.ones_like(mean)/mean.shape[1]
    
    cost1 = torch.lgamma(target_precision + epsilon)-torch.lgamma(precision + epsilon)
    cost2= torch.sum(torch.lgamma(mean * precision + epsilon) - torch.lgamma(target_mean * target_precision + epsilon), axis=1)
    cost3= torch.sum((target_precision * target_mean - mean * precision) * (torch.digamma(target_mean * target_precision + epsilon) 
                                                                       -torch.digamma(target_precision + epsilon)), axis=1)

    cost = torch.mean(cost1+cost2+cost3)
    
    #cost=torch.mean((precision-target_precision)**2)
    return cost
#选择ACE，即考虑每个预测类别概率是否校准并自适应bin
def cal_loss(target_mean, mean,bins=10):
    N=mean.shape[0]
    K=mean.shape[1]
    ace=0
    nu=N//bins
    for i in range(K):
        con=mean[:,i]
        lab=target_mean[:,i]
   
        con_,index=torch.sort(con)
        lab_=lab[index]
        for j in range(bins):
            if j==bins-1:
                con_hat=torch.mean(con_[j*nu:])
                miu_hat=torch.mean(lab_[j*nu:])
                ace=ace+torch.abs(con_hat-miu_hat)
            
            elif j<bins-1:
                con_hat=torch.mean(con_[j*nu:(j+1)*nu])
                miu_hat=torch.mean(lab_[j*nu:(j+1)*nu])
                ace=ace+torch.abs(con_hat-miu_hat)

    return ace/(K*bins)

def fit_model(model,nb_epoch, X, Y,X_val,Y_val,tra_loss,val_loss,tra_acc,val_acc,tra_cal,val_cal,min_loss,p_,lete):
    optimizer= torch.optim.Adam(params=model.parameters(),lr=0.005)  

    x_val=Variable(torch.FloatTensor(X_val)).cuda()
    y_val=Variable(torch.FloatTensor(Y_val)).cuda()
    N = X.shape[0]
    if lete!=0:
        model.eval()
        with torch.no_grad():
            mean_val, alpha_val,regularization_val,p2_val=model(x_val)
        min_loss=acc_loss(y_val, mean_val)+l1*KL_ID( y_val, mean_val, alpha_val)+l2*cal_loss(y_val, mean_val)+ l3*regularization_val
        best=model
    model.cuda()   
    for i in range(nb_epoch):
        index_t=[i for i in range(N)]
        np.random.shuffle(index_t)
        X=X[index_t]
        Y=Y[index_t]
        loss_tra=0
        calibration_tra=0
        accuracy_tra=0
        model.train()
        print("epoch:%d/%d>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"%(i+1,nb_epoch))
        for batch in range(int(np.ceil(X.shape[0]/batch_size))):
            batch = (batch + 1)
            if batch==int(np.ceil(X.shape[0]/batch_size)):
                _x = X[batch_size*(batch-1):]
                _y = Y[batch_size*(batch-1):]
            else:
                _x = X[batch_size*(batch-1): batch_size*batch]
                _y = Y[batch_size*(batch-1): batch_size*batch]
            
            x = Variable(torch.FloatTensor(_x)).cuda()
            y = Variable(torch.FloatTensor(_y)).cuda()

            mean, alpha,regularization,p2= model(x)
            
            loss = acc_loss(y, mean)+l1*KL_ID( y, mean, alpha)+ l2*cal_loss(y, mean)+ l3*regularization
            calibration_loss=cal_loss(y, mean).item()
            accuracy=acc_(y,mean).item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_tra += loss.item()*_x.shape[0]
            calibration_tra +=calibration_loss*_x.shape[0]
            accuracy_tra +=accuracy*_x.shape[0]
            print("\r %d/%d,loss: %f,ACE:%f,ACC:%f"%(batch,int(np.ceil(X.shape[0]/batch_size)),loss.item(),calibration_loss,accuracy),end='')
        print("\n")
        print("training:%f,  %f,  %f"%(loss_tra/N,calibration_tra/N,accuracy_tra/N))
        print("dropout_rate:\n",p2)
        tra_loss.append(loss_tra/N)
        tra_acc.append(accuracy_tra/N)
        tra_cal.append(calibration_tra/N)
        p_.append(p2)
        
        model.eval()
        with torch.no_grad():
            mean_val, alpha_val,regularization_val,p2_val=model(x_val)
        
        loss_val = acc_loss(y_val, mean_val)+l1*KL_ID( y_val, mean_val, alpha_val)+ l2*cal_loss(y_val, mean_val)+ l3*regularization_val
        val_loss.append(loss_val)
        
        if loss_val<min_loss:
            min_loss=loss_val
            print("copy model")
            best=copy.deepcopy(model)
        n_ep=i+nb_epoch*lete
        calibration_val=cal_loss(y_val, mean_val).item()
        accuracy_val=acc_(y_val,mean_val).item()
        val_acc.append(accuracy_val)
        val_cal.append(calibration_val)
        print("val:%f,  %f,  %f"%(loss_val,calibration_val,accuracy_val))
    return best,tra_loss,val_loss,tra_acc,val_acc,tra_cal,val_cal,n_ep


def fit_model_OOD(model,nb_epoch, X, X_val,tra_loss,val_loss,min_loss,p_,lete):
    optimizer1= torch.optim.Adam(params=model.parameters(),lr=0.0001) 
    x_val=Variable(torch.FloatTensor(X_val)).cuda()
    N = X.shape[0]
    
    if lete!=-1:
        model.eval()
        with torch.no_grad():
            mean_val, alpha_val,regularization_val,p2_val=model(x_val)
        min_loss=l4*KL_OD(mean_val, alpha_val)
        best=model
    model.cuda()
    for i in range(nb_epoch):
        index_t=[i for i in range(N)]
        np.random.shuffle(index_t)
        X=X[index_t]
        loss_tra=0
        alpha_tra=0
        model.train()
        print("epoch:%d/%d>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"%(i+1,nb_epoch))
        for batch in range(int(np.ceil(X.shape[0]/batch_size))):
            batch = (batch + 1)
            if batch==int(np.ceil(X.shape[0]/batch_size)):
                _x = X[batch_size*(batch-1):]
            else:
                _x = X[batch_size*(batch-1): batch_size*batch]
                
            x = Variable(torch.FloatTensor(_x)).cuda()

            mean, alpha,regularization,p2= model(x)
            
            loss = l4*KL_OD(mean, alpha)
            
            
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            
            loss_tra += loss.item()*_x.shape[0]
            alpha_tra +=alpha.sum()
            print("\r %d/%d,loss: %f,alpha:%f"%(batch,int(np.ceil(X.shape[0]/batch_size)),loss.item(),alpha.mean()),end='')
        print("\n")
        print("training:%f,  %f"%(loss_tra/N,alpha_tra/N))
        print("dropout_rate:\n",p2)
        tra_loss.append(loss_tra/N)
        p_.append(p2)
        
        model.eval()
        with torch.no_grad():
            mean_val, alpha_val,regularization_val,p2_val=model(x_val)
        
        loss_val = l4*KL_OD(mean_val, alpha_val)
        val_loss.append(loss_val)
        if loss_val<min_loss:
            min_loss=loss_val
            print("copy model")
            best=copy.deepcopy(model)
        n_ep=i+nb_epoch*lete

        print("val:%f,  %f"%(loss_val,alpha_val.mean()))
    return best,tra_loss,val_loss,alpha_tra,alpha_val.mean(),n_ep


nb_epoch =5
nb_epoch1 =5
batch_size = 500
hiddensize=[x_train_sample.shape[2],128,64,32]
l = 0.1 # Lengthscale

N = x_train_sample.shape[0]
wr = l**2. / N
br=1/N
dr = 2. / N
model= Model(wr,br,dr,hiddensize).to(device) 
l1=0.001    #KL_ID
l2=0.0002   #calibration
l3=0.1      #regularization
l4=0.001      #KL_OD

min_loss=1e6
min_loss1=1e6
val_loss=[]
tra_loss=[]
val_loss1=[]
tra_loss1=[]
tra_acc=[]
val_acc=[]
tra_cal=[]
val_cal=[]
p_=[]
lete=0

X_train, Y_train = x_train_sample, y_train_sample
X_val, Y_val =x_val_sample,y_val_sample


X_train_OD = x_train_sample_OD
X_val_OD =x_val_sample_OD
for i in range(20):
    model,tra_loss,val_loss,tra_acc,val_acc,tra_cal,val_cal,n_ep= fit_model(model,nb_epoch, X_train, Y_train,X_val, Y_val,tra_loss,val_loss,tra_acc,val_acc,tra_cal,val_cal,min_loss,p_,lete)
    
    model,tra_loss1,val_loss1,tra_alpha,val_alpha,n_ep= fit_model_OOD(model,nb_epoch1, X_train_OD,X_val_OD,tra_loss1,val_loss1,min_loss1,p_,lete)
    lete=lete+1
model,tra_loss,val_loss,tra_acc,val_acc,tra_cal,val_cal,n_ep= fit_model(model,nb_epoch, X_train, Y_train,X_val, Y_val,tra_loss,val_loss,tra_acc,val_acc,tra_cal,val_cal,min_loss,p_,lete)


ps_dense1 = np.array([torch.sigmoid(module.p_logit_dense).cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'p_logit_dense')])
ps = np.array([torch.sigmoid(module.p_logit).cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'p_logit')])
ps_rec = np.array([torch.sigmoid(module.p_logit_rec).cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'p_logit_rec')])


#loss图
plot_.loss_plot(tra_loss,val_loss)
plot_.loss_plot(tra_loss1,val_loss1)
plot_.acc_plot(tra_acc, val_acc)
plot_.ace_plot(tra_cal, val_cal)


#%%预测
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)  
SimSun = FontProperties(fname='C:\WINDOWS\Fonts\SIMSUN.TTC')  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
Times_New_Roman = FontProperties(fname='C:\WINDOWS\Fonts\TIMES.TTF')


import scipy
import seaborn as sns
from scipy.stats import dirichlet
from scipy.special import digamma,gamma
def dis_entr(alpha_o,means):
    alpha_c=means*alpha_o
    '''
    en=np.mean(np.sum(np.log(gamma(alpha_c))-np.log(gamma(alpha_o)),2)
               -np.sum((alpha_c-1)*(digamma(alpha_c)-digamma(alpha_o)),2),0)
    '''
    en=[np.mean([dirichlet(alpha_c[i,j]).entropy() for i in range(K_test)])for j in range(means.shape[1])]
    return en
#dirichlet([1,1,1,1,1,1,1]).entropy()
def pre_entr(means):
    en=-np.sum(means*np.log(means),1)
    return en
#distributional uncertainty 纳入aleatoric 
def ale_entr(means):
    en=-np.mean(np.sum(means*np.log(means),2),0)
    return en
#distributional uncertainty 纳入epistemic
def ale_entr_single(pc_,alpha_0):
    alpha=pc_*alpha_0
    en=[]
    for j in range(alpha.shape[1]):
        miu_=np.array([np.array([np.random.dirichlet(alpha[i,j], size=None)for _ in range(10)])for i in range(alpha.shape[0])])
        en.append(-np.mean(np.mean(np.sum(miu_*np.log(miu_),2),1),0))
    return np.array(en)
def ACE(target_mean, mean,bins=10):
    N=mean.shape[0]
    K=mean.shape[1]
    con_hat=np.zeros((K,bins))
    miu_hat=np.zeros((K,bins))
    nu=N//bins
    for i in range(K):
        con=mean[:,i]
        lab=target_mean[:,i]
   
        con_=np.sort(con)
        index=np.argsort(con)
        lab_=lab[index]
      
        for j in range(bins):
            if j==bins-1:
                con_hat[i,j]=np.mean(con_[j*nu:])
                miu_hat[i,j]=np.mean(lab_[j*nu:])
            
            elif j<bins-1:
                con_hat[i,j]=np.mean(con_[j*nu:(j+1)*nu])
                miu_hat[i,j]=np.mean(lab_[j*nu:(j+1)*nu])
    ace=np.mean(np.abs(con_hat-miu_hat))

    return ace,con_hat,miu_hat

def SCE(target_mean, mean,bins=10):
    N=mean.shape[0]
    K=mean.shape[1]
    con_hat=np.zeros((K,bins))
    miu_hat=np.zeros((K,bins))
    px=np.linspace(0,1,bins+1)
    nu=np.zeros((K,bins))
    for i in range(K):
        con=mean[:,i]
        lab=target_mean[:,i]
        for j in range(bins):
            for k in range(N):
                if con[k]>px[j] and con[k]<=px[j+1]:
                    con_hat[i,j]+=con[k]
                    miu_hat[i,j]+=lab[k]
                    nu[i,j]+=1
            if nu[i,j]!=0:
                miu_hat[i,j]=miu_hat[i,j]/nu[i,j] 
                con_hat[i,j]=con_hat[i,j]/nu[i,j]
            else:
                miu_hat[i,j]=np.nan
                con_hat[i,j]=np.nan
    
    ace=np.mean(np.abs(con_hat-miu_hat))
    return ace,con_hat,miu_hat
def ECE(target_mean, mean,bins=10):
    true_=np.argmax(target_mean,1)
    pred=np.argmax(mean,1)
    p=np.max(mean,1)
    px=np.linspace(0,1,bins+1)
    con_hat=np.zeros((bins,))
    miu_hat=np.zeros((bins,))
    nu=np.zeros((bins,))
    n=0
    ece=0
    for i in range(bins):
        for j in range(mean.shape[0]):
            if p[j]>px[i] and p[j]<=px[i+1]:
                con_hat[i]+=p[j]
                nu[i]+=1
                if pred[j]==true_[j]:
                    miu_hat[i]+=1
        if nu[i]!=0:
            miu_hat[i]=miu_hat[i]/nu[i] 
            con_hat[i]=con_hat[i]/nu[i]
            ece+=np.abs(miu_hat[i]-con_hat[i])
            n=n+1
        else:
            miu_hat[i]=np.nan
            con_hat[i]=np.nan
    ece=ece/n
    return ece
def acc_t(target_mean, mean):
    pred=mean.argmax(1)
    tru=target_mean.argmax(1)
    num_correct=0
    for i in range(mean.shape[0]):
        if pred[i]==tru[i]:
            num_correct += 1

    return num_correct/mean.shape[0]
#健康类别的精确率
def prec_(target_mean, mean):
    pred=mean.argmax(1)
    tru=target_mean.argmax(1)
    num_correct=0
    num_=0
    for i in range(mean.shape[0]):
        if pred[i]==0:
            num_ += 1
            if tru[i]==0:
                num_correct+=1
    return num_correct/num_
def hd(target_mean, mean):
    pred=mean.argmax(1)
    tru=target_mean.argmax(1)
    tp=[]
    fp=[]
    t_=[]
    f_=[]
    for i in range(mean.shape[0]):
        if pred[i]==tru[i]:
            tp.append(pre_entr(mean[i:i+1]))
            t_.append(i)
        else:
            fp.append(pre_entr(mean[i:i+1]))
            f_.append(i)
    return -np.mean(tp)+np.mean(fp),t_,f_


X_test, Y_test =x_test_sample,y_test_sample
X_test_OD=x_test_sample_OD
T1=time.time()
model.eval()
K_test=100
with torch.no_grad():
    MC_rul=[(model(Variable(torch.FloatTensor(X_test)).cuda())) for _ in range(K_test)]
    MC_rul_OD=[(model(Variable(torch.FloatTensor(X_test_OD)).cuda())) for _ in range(K_test)]
T2=time.time()   
print(T2-T1)
pc_ = torch.stack([tup[0] for tup in MC_rul]).view(K_test, X_test.shape[0],Y_test.shape[1]).cpu().data.numpy() 
alpha_0 = torch.stack([tup[1] for tup in MC_rul]).view(K_test, X_test.shape[0],1).cpu().data.numpy() 
means_=np.mean(pc_,0)

pc_OD = torch.stack([tup[0] for tup in MC_rul_OD]).view(K_test, X_test_OD.shape[0],Y_test.shape[1]).cpu().data.numpy() 
alpha_0_OD = torch.stack([tup[1] for tup in MC_rul_OD]).view(K_test, X_test_OD.shape[0],1).cpu().data.numpy() 
means_OD=np.mean(pc_OD,0)

acc_test=acc_t(Y_test,means_)
ace_test,con_test,miu_test=ACE(Y_test,means_)
acee=np.mean(np.abs(con_test-miu_test),1)
preci=prec_(Y_test, means_)
ece=ECE(Y_test,means_)
tfp,t_,f_=hd(Y_test,means_)
#ace_test,con_test,miu_test=ECE(Y_test,means_)

pre_uncertainty=pre_entr(means_)
alea_uncertainty=ale_entr_single(pc_,alpha_0)

dist_uncertainty=dis_entr(alpha_0,pc_)
epis_uncertainty=pre_uncertainty-alea_uncertainty


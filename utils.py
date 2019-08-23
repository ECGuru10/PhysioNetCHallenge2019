import numpy as np
import time
import os
import torch

def normalization(data,minv,maxv,q,w):
    
    
    data_new=[]
    for k,pac in enumerate(data):
        tmp=q+(w-q)*(pac-minv)/(maxv-minv)
        data_new.append(tmp)
        
    return data_new



def replace_nan(data,q):
    data_new=[]
    for k,pac in enumerate(data):
        tmp=pac
        tmp[np.isnan(pac)]=q
        data_new.append(tmp)
        
    return data_new





class AdjustLearningRate():
    def __init__(self,lr_step=10):
        self.lr_step=lr_step
        self.best_loss=999999999
        self.best_loss_pos=0
        self.stopcount=0
        
    def step(self,optimizer,iteration=0,loss=0):

        try:
            with open('lr_change.txt', 'r') as f:
                x = f.readlines()
                lr=float(x[0])
            time.sleep(1)
            os.remove('lr_change.txt')
            
            print('lr was set to: ' + str(x))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        except:
            pass
        
        if  0:#loss<self.best_loss:
            self.best_loss_pos=iteration
            self.best_loss=loss
#        
        print(self.best_loss,loss)
#        
#        print(str(self.batch*iteration-self.best_loss_pos) + '////' + str(self.lr_step))
        if  (iteration-self.best_loss_pos)>self.lr_step:
            self.stopcount+=1
            self.best_loss_pos=iteration
            self.best_loss=loss
            print('lr down')
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] *0.3
                
        if  self.stopcount>=4:
            return 1
        else:
            return 0

def dice_loss(prob, target, eps=1e-7):
    
    neg_prob = 1 - prob
    probas = torch.stack([prob, neg_prob], dim=1)
    
    neg_target=1-target
    targets=torch.stack([target, neg_target], dim=1)
    
    intersection = torch.sum(probas * targets,dim=0)
    cardinality = torch.sum(probas + targets,dim=0)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)
    
    


def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)




def compute_prediction_utility_muj(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0, check_errors=True):

    
    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    
    t=np.arrange(n)
    
    tt=t <= (t_sepsis + dt_late)
    
    if is_septic:
        
        pass
    
    else:
    
        pass
    
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)




    
def get_utility(x,preds,lbls):
    
    dt_early   = -12
    dt_optimal = -6
    dt_late    = 3

    max_u_tp = 1
    min_u_fn = -2
    u_fp     = -0.05
    u_tn     = 0
    
    num_files=len(preds)
    
    observed_utilities = np.zeros(num_files)
    best_utilities     = np.zeros(num_files)
    worst_utilities    = np.zeros(num_files)
    inaction_utilities = np.zeros(num_files)

    for k in range(num_files):
        lbl = lbls[k]
        pred = preds[k]>x
        
        num_rows  = len(lbl)
        
        best_predictions     = np.zeros(num_rows)
        worst_predictions    = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)

        if np.any(lbl):
            t_sepsis = np.argmax(lbl) - dt_optimal
            best_predictions[max(0, t_sepsis + dt_early) : min(t_sepsis + dt_late + 1, num_rows)] = 1
        worst_predictions = 1 - best_predictions

        observed_utilities[k] = compute_prediction_utility(lbl, pred, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        best_utilities[k]     = compute_prediction_utility(lbl, best_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        worst_utilities[k]    = compute_prediction_utility(lbl, worst_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_utilities[k] = compute_prediction_utility(lbl, inaction_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility     = np.sum(best_utilities)
    unnormalized_worst_utility    = np.sum(worst_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)

    normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)

    return normalized_observed_utility




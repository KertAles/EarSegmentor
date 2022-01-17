import math
import numpy as np
import matplotlib.pyplot as plt

class Evaluation:
    
    def compute_rank1(self, dist_arr, gt):
        count_all = 0
        count_correct = 0
        
        for idx, dist in enumerate(dist_arr) :
            count_all += 1
            count_correct += self.compute_rank1_loc(dist, gt[idx])
            
        return count_correct / count_all
    
    
    def compute_rank1_loc(self, dist_arr, gt):
        is_top = 0
        
        if gt == dist_arr[0][1] :
            is_top = 1
                
        return is_top
    
    def compute_rankn(self, dist_arr, gt, n):
        count_all = 0
        count_correct = 0
        
        for idx, dist in enumerate(dist_arr) :
            count_all += 1
            count_correct += self.compute_rankn_loc(dist, gt[idx], n)
            
        return count_correct / count_all

    def compute_rankn_loc(self, dist_arr, gt, n):
        
        in_topn = 0
        
        for i in range(min(n, len(dist_arr))) :
            
            if gt == dist_arr[i][1] :
                in_topn = 1
                
        return in_topn
    
    def compute_cmc(self, dist_arr, gt) :
        cmc_vect = []
        
        for i in range(len(dist_arr[0])) :
            cmc_vect.append(self.compute_rankn(dist_arr, gt, i+1))
            
        return cmc_vect
    
    
    def compute_display_cmc(self, dist_arr, gt) :
        cmc_vect = self.compute_cmc(dist_arr, gt)
        
        line = np.array(list(range(len(dist_arr[0])))) / len(dist_arr[0])
        
        plt.title('CMC plot')
        plt.ylabel('Recognition %')
        plt.xlabel('Rank')
        plt.plot(cmc_vect, 'g', label='prediction rank')
        plt.plot(line, 'r', label='chance')
        plt.legend()
        
        
    def compute_cmc_auc(self, dist_arr, gt) :
        cmc_vect = self.compute_cmc(dist_arr, gt)
        
        
        
 
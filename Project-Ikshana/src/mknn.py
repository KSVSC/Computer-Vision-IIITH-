import numpy as np
import math

class MKNN:
    def __init__(self, X,y):
        self.X = X
        self.y = y

    def find_distance(self, pt1, pt2):
        distance=0
        for i in range(0,len(pt1)):
            distance=distance+pow(pt1[i]-pt2[i],2)
        
        return math.sqrt(distance)

    def get_freq(self,neigh_label):
        class_dict={}
        for i in range(0,len(neigh_label)):
            if i not in class_dict:
                class_dict[i]=1
            else:
                class_dict[i]=class_dict[i]+1
        
        return class_dict

    def get_neighbours(self,test_pt,k):
        distances=[]
        neigh_label=[]
        neigh=[]
        for (train_pt,label) in zip(self.X,self.y):
            dist=self.find_distance(test_pt,train_pt)
            distances.append((train_pt,dist,label))
        distances.sort(key=lambda tup: tup[1])
        for i in range(0,k):
            neigh.append(distances[i][0])
            neigh_label.append(distances[i][2])

        return neigh,neigh_label

    def k_nearest_neighbours(self,X_test,k):
        all_class_dict=[]
        all_neigh_label=[]
        all_neigh=[]
        for pt in X_test:
            neigh,neigh_label=self.get_neighbours(pt,k)
            class_dict=self.get_freq(neigh_label)
            all_neigh.append(neigh)
            all_neigh_label.append(neigh_label)
            all_class_dict.append(class_dict)

        return all_class_dict,all_neigh,all_neigh_label
    
    def marginalized_knn(self,X_test,k):
        all_class_dictA,all_neighA,all_neigh_labelA=self.k_nearest_neighbours(X_test,k)
        predicted_label=[]
        predicted_prob=[]
        for class_dictA,neighA,neigh_labelA in zip(all_class_dictA,all_neighA,all_neigh_labelA):
            all_class_dictB, all_neighB, all_neigh_labelB = self.k_nearest_neighbours(neighA,k)
            pairs_prob=[]
            keysA=class_dictA.keys()
            for class_dictB in all_class_dictB:
                sum1=0
                for i in keysA:
                    if i in class_dictB:
                        sum1=sum1+(class_dictA.get(i)*class_dictB.get(i))
                prob=sum1/(k*k)
                pairs_prob.append(prob)

            max_idx = np.argmax(pairs_prob)
            predicted_prob.append(pairs_prob[max_idx])
            predicted_label.append(neigh_labelA[max_idx])
       
        return predicted_label,predicted_prob

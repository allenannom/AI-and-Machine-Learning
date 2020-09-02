# -*- coding: utf-8 -*-
"""
@author: Allen Annom
"""
import numpy as np
import matplotlib.pyplot as plt
from tree import mean_tree_score
from forest import mean_forest_score

#print gridsearchh result not cross validation
print("THE MEAN TREE SCORE IS %0.2f" % mean_tree_score)
print("THE MEAN FOREST SCORE IS %0.2f" % mean_forest_score)

objects = ('Decision Tree', 'Random Forest')
y_pos = np.arange(len(objects))
performance = [mean_tree_score,mean_forest_score]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Score')
plt.title('Algorithm Scores')
 
plt.show()

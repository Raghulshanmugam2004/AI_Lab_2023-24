# Ex.No: 3  Implementation of Minimax Search
### DATE: 05/11/2023                                                                           
### REGISTER NUMBER :  212222040128
### AIM: 
Write a mini-max search algorithm to find the optimal value of MAX Player from the given graph.
### Algorithm:
1. Start the program
2. import the math package
3. Specify the score value of leaf nodes and find the depth of binary tree from leaf nodes.
4. Define the minimax function
5. If maximum depth is reached then get the score value of leaf node.
6. Max player find the maximum value by calling the minmax function recursively.
7. Min player find the minimum value by calling the minmax function recursively.
8. Call the minimax function  and print the optimum value of Max player.
9. Stop the program. 
10.end.

### Program:
```
import math
def minimax (curDepth, nodeIndex,maxTurn, scores,targetDepth):

    if (curDepth == targetDepth):
        return scores[nodeIndex]
    if (maxTurn):
        return max(minimax(curDepth + 1, nodeIndex * 2,False, scores, targetDepth),minimax(curDepth + 1, nodeIndex * 2 + 1,False, scores, targetDepth))
    else:
        return min(minimax(curDepth + 1, nodeIndex * 2,True, scores, targetDepth),minimax(curDepth + 1, nodeIndex * 2 + 1,True, scores, targetDepth))

scores = [2,3,4,5,-1,4,2,6]
treeDepth = math.log(len(scores), 2)
print("The optimal value is : ", end = "")
print(minimax(0, 0, True, scores, treeDepth))
```
### Output:

![image](https://github.com/Raghulshanmugam2004/AI_Lab_2023-24/assets/119561118/7641b76f-ce89-41a4-af46-563dda3f3222)

### Result:
Thus the optimum value of max player was found using minimax search.

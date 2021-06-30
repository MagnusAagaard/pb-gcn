from .graph import Graph

num_node = 17

inward_ori_index = [(17,15), (16,14), (15,13), (14,12), (13,7), (12,6), (11,9), (10,8), (9,7), (8,6), (5,7), (4,6), (3,5), (2,4), (1,2), (1,3)]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

head = [(0,1), (0,2), (1,3), (2,4), (3,5), (4,6)]
lefthand = [(5,7), (7,9)]
righthand = [(6,8), (8,10)]
hands = lefthand + righthand
torso = [(5,6), (5,11), (6,12), (11,12)]
leftleg = [(11,13), (13,15)]
rightleg = [(12,14), (14,16)]
legs = leftleg + rightleg

class CenterNetGraph(Graph):
    def __init__(self,
                 labeling_mode='uniform'):
        super(CenterNetGraph, self).__init__(num_node=num_node,
                                      inward=inward,
                                      outward=outward,
                                      parts=[head, hands, torso, legs],
                                      labeling_mode=labeling_mode)

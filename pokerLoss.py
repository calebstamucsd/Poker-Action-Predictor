import torch.nn as nn
import torch

# class fourLoss(nn.Module):
#     def __init__(self, loss_type):
#         super(fourLoss, self).__init__()
#         self.loss_type = loss_type

#     def forward(self, y, t):
#         '''
#         y: model generated output shape: (104) i.e. [card1 + card2] where each element is one_hot values
#         t: target value shape: (104) i.e. [card1 + card2] where each element is one_hot values
#         '''
#         ce = nn.CrossEntropyLoss()
#         yc1, yc2 = torch.split(y, 52, dim=1)
#         tc1, tc2 = torch.split(t, 52, dim=1)
#         loss1 = torch.mean(ce(yc1, tc1), ce(yc2, tc2))
#         loss2 = torch.mean(ce(yc2, tc1), ce(yc1, tc2))
#         return torch.minimum(loss1, loss2)

def fourLoss(y, t):
    ce = nn.CrossEntropyLoss()
    yc1, yc2 = torch.split(y, 52, dim=1)
    tc1, tc2 = torch.split(t, 52, dim=1)
    loss1 = (ce(yc1, tc1) + ce(yc2, tc2)) / 2
    loss2 = (ce(yc2, tc1) + ce(yc1, tc2)) / 2
    return torch.minimum(loss1, loss2)



        # '''
        # y: model generated output shape: (34) i.e. [rank1 + suit1 + rank2 + suit2] where each element is a one-hot encoded representation of a card
        # t: target value shape: (34) i.e. [rank1 + suit1 + rank2 + suit2] where each element is a one-hot encoded representation of a card
        # '''
        # if self.loss_type == "loss 1":

        #     y1_r, y1_s = y[0:13], y[13:17]
        #     y2_r, y2_s = y[17:30], y[30:34]
        #     t1_r, t1_s = t[0:13], t[13:17]
        #     t2_r, t2_s = t[17:30], t[30:34]

        #     ce = nn.CrossEntropyLoss()
            
        #     loss1 = ce(y1_r, t1_r) + ce(y1_s, t1_s) + ce(y2_r, t2_r) + ce(y2_s, t2_s)
        #     loss2 = ce(y2_r, t1_r) + ce(y2_s, t1_s) + ce(y1_r, t2_r) + ce(y1_s, t2_s)

        #     return min(loss1/4, loss2/4)
    
def accuracy(y, t):
    yc1, yc2 = torch.argmax(y[:52]), torch.argmax(y[52:])
    tc1, tc2 = torch.argmax(t[:52]), torch.argmax(t[52:])
    accuracy_1 = int((yc1 == tc1) and (yc2 == tc2))
    accuracy_2 = int((yc1 == tc2) and (yc2 == tc1))
    return max(accuracy_1, accuracy_2)

    
#testing

# s1 = [0, 0.5, 0.5, 0]
# s2 = [0, 0, 1, 0]
# s_T1 = [0, 1, 0, 0]
# s_T2 = [0, 0, 1, 0]

# r1 = [0, 0, 0, 0.8, 0, 0.2, 0, 0, 0, 0, 0, 0, 0]
# r2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
# r_T1 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# r_T2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# y = torch.as_tensor(r1+s1+r2+s2).type(torch.float32)
# t = torch.as_tensor(r_T1+s_T1+r_T2+s_T2).type(torch.float32)

# ce = torch.nn.CrossEntropyLoss()
# loss = ce(t, s_T1)
# print(loss)
# loss.backward()

# loss = fourLoss(y, t)
# print(loss)
# print(loss.item())
# loss.backward()


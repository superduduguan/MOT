!pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl 
import torch
import torch.nn as nn
import torch.nn.init as init
#is_cuda = True
#bidirection = False

class DHN(nn.Module):#初始化/bidirection
    def __init__(self, element_dim, hidden_dim, target_size, minibatch, is_train=True):
        super(DHN, self).__init__()
        self.hidden_dim = hidden_dim
        self.minibatch = minibatch
        self.is_train = is_train

        self.lstm_row = nn.GRU(input_size=element_dim, hidden_size=hidden_dim, num_layers=2, bias=True, batch_first=False, dropout=0)
        self.lstm_col = nn.GRU(512, hidden_dim, num_layers=2)
        self.hidden2tag_1 = nn.Linear(hidden_dim, target_size)
        self.hidden_row = self.init_hidden(1)
        self.hidden_col = self.init_hidden(1)

        init_GRU()

    def init_GRU(self):
        if self.is_train == True:
            for m in self.modules():#查看每一层
                if isinstance(m, nn.GRU):#如果是GRU层
                    print("weight and bias initialization")#For nn.LSTM, there are four associated parameters ih/jj/weight/bias
                    init.xavier_uniform_(m.weight_ih_l0.data)#正交初始化
                    init.xavier_uniform_(m.weight_hh_l0.data)#input-hidden-first
                    init.xavier_uniform_(m.weight_ih_l0_reverse.data)
                    init.xavier_uniform_(m.weight_hh_l0_reverse.data)
                    init.xavier_uniform_(m.weight_ih_l1.data)
                    init.xavier_uniform_(m.weight_hh_l1.data)
                    init.xavier_uniform_(m.weight_ih_l1_reverse.data)
                    init.xavier_uniform_(m.weight_hh_l1_reverse.data)
                    m.bias_ih_l0.data[0:self.hidden_dim].fill_(-1)# initial gate bias as one
                    m.bias_hh_l0.data[0:self.hidden_dim].fill_(-1)
                    m.bias_ih_l1.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l1.data[0:self.hidden_dim].fill_(-1)
                    m.bias_ih_l0_reverse.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l0_reverse.data[0:self.hidden_dim].fill_(-1)
                    m.bias_ih_l1_reverse.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l1_reverse.data[0:self.hidden_dim].fill_(-1)
    def init_hidden(self, batch):#隐层全部置零
        hidden = (torch.zeros(1, batch, self.hidden_dim).cuda(), torch.zeros(1, batch, self.hidden_dim).cuda())
        return hidden

    def forward(self, Dt):# Dt is of shape [batch, h, w]
        input_row = Dt.view(Dt.size(0), -1, 1).permute(1, 0, 2).contiguous()#input_row is of shape [h*w, batch, 1]。。torch.view = resize， torch.size(0)就是第一维的大小：batch，torch.permute():重新排序
        lstm_R_out, self.hidden_row = self.lstm_row(input_row, self.hidden_row)
        lstm_R_out = lstm_R_out.view(-1, lstm_R_out.size(2))
        lstm_R_out = lstm_R_out.view(Dt.size(1), Dt.size(2), Dt.size(0), -1)
        input_col = lstm_R_out.permute(1, 0, 2, 3).contiguous()
        input_col = input_col.view(-1, input_col.size(2), input_col.size(3)).contiguous()
        lstm_C_out, self.hidden_col = self.lstm_col(input_col, self.hidden_col)
        lstm_C_out = lstm_C_out.view(Dt.size(2), Dt.size(1), Dt.size(0), -1).permute(1, 0, 2, 3).contiguous()
        lstm_C_out = lstm_C_out.view(-1, lstm_C_out.size(3))
        # [h*w, batch, 1]
        tag_space = self.hidden2tag_1(lstm_C_out)
        tag_space = self.hidden2tag_2(tag_space)
        tag_space = self.hidden2tag_3(tag_space).view(-1, Dt.size(0))
        tag_scores = torch.sigmoid(tag_space)
        # tag_scores is of shape [batch, h, w] as Dt
        return tag_scores.view(Dt.size(1), Dt.size(2), -1).permute(2, 0, 1).contiguous()
print('x')

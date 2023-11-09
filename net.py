import opensim as osim
import torch
import numpy as np
from tianshou.utils.net.common import Net


def split_batch(batch, n=1):
    # l = batch.shape[1]
    l = len(batch[0])
    for ndx in range(0, l, n):
        # yield batch[:, ndx:min(ndx + n, l)]
        yield [batch[i][ndx:min(ndx + n, l)] for i in range(len(batch))]


class TestNet:
    def __init__(self, input_shape, output_shape, hidden_sizes, update_num=1):
        self.net = Net(state_shape=input_shape, action_shape=output_shape, hidden_sizes=hidden_sizes, activation=torch.nn.Tanh)
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)
        self.net_optim = torch.optim.Adam(self.net.parameters(), lr=5e-3)
        self.update_num = update_num

    def init(self, input, output, iter):
        losses = []
        errors = []
        # i = [input for _ in range(self.update_num)]
        # o = [output for _ in range(self.update_num)]
        i = [input for _ in range(5)]
        o = [output for _ in range(5)]
        # data = torch.tensor([i, o])
        data = [i, o]
        for _ in range(iter):
            loss, error = self.update_init(data)
            losses.append(loss)
            errors.append(error)
        return losses, errors

    def update_init(self, batch):
        losses = []
        errors = []
        # bsz = len(b) // self.update_num
        for b in split_batch(batch, n=self.update_num):
        # for b in split_batch(batch, n=2):
            logits = self.net(b[0])
            # print(logits[0], b[1])
            loss = -torch.nn.functional.logsigmoid(logits[0] - torch.tensor(b[1])).mean()
            # loss = -(logits[0] - torch.tensor(b[1])).mean()
            self.net_optim.zero_grad()
            loss.backward()
            self.net_optim.step()
            error = np.asarray(logits[0].detach().numpy() - b[1])[0]
            losses.append(loss.item())
            errors.append(error)
        return losses, errors

    def update(self, input, batch):
        losses = []
        errors = []
        # bsz = len(b) // self.update_num
        for b in split_batch(batch, n=self.update_num):
            i = torch.tensor([input for _ in range(b[0].shape[0])])
            output = self.net(i)
            o = output[0].detach().numpy()
            # loss1 = -torch.nn.functional.logsigmoid(b[0] - b[1]).mean()
            error = np.asarray(b[0] - b[1])[0]
            loss = torch.nn.functional.logsigmoid((b[0] * torch.tensor(o) / output[0] - b[1]) * 100).mean()
            self.net_optim.zero_grad()
            loss.backward()
            self.net_optim.step()
            losses.append(loss.item())
            errors.append(error)
        return losses, errors
    # def update(self, batch):
    #     losses = []
    #     # bsz = len(b) // self.update_num
    #     for b in split_batch(batch, n=self.update_num):
    #         logits = self.net(b[0])
    #         # loss = -torch.nn.functional.logsigmoid(torch.mean(logits[0] - b[1])).mean()
    #         loss = -torch.nn.functional.logsigmoid(logits[0] - b[1]).mean()
    #         self.net_optim.zero_grad()
    #         loss.backward()
    #         self.net_optim.step()
    #         losses.append(loss.item())
    #     return losses


if __name__ == '__main__':
    i = [[1,2,3] for _ in range(10)]
    o = [[1,2,3] for _ in range(10)]
    d = torch.tensor([i, o])

    for b in split_batch(d, 4):
        print(b[0])


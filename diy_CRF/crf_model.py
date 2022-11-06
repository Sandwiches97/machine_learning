# -*- coding: utf-8 -*-
import torch
from torch import nn
from typing import Dict
from transformers import AutoModel, AutoTokenizer

def sequence_mask(lens: torch.Tensor, max_len: int=None)->torch.Tensor:
    """ Generate a sequence mask tensor from sequence lengths, used by CRF

    :param lens:    (batch, max_seq_len)
    :param max_len: int
    :return:        (batch, max_seq_len)
    """
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask

class CRF(nn.Module):
    def __init__(self, label_vocab: Dict[str, int], bioes: bool=False):
        """

        :param label_vocab: 字典，key=label，value=对应的idx
            例如，{"O": 0, "B-PER": 1, ...}
        :param bioes: 两种模式的标注：bioes，bio
            {B-begin 开始, I-inside 中间, O-outside 其他, E-end 结尾, S-single 单个字符}
        """
        super(CRF, self).__init__()
        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2      # 加上了 <sos>，<eos>
        self.bioes = bioes

        self.start = self.label_size-2              # <sos> 的label
        self.end = self.label_size-1                # <eos> 的label
        # 初始化一个传输矩阵 (状态个数，状态个数)
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        """ 初始化转移矩阵
        1. 将所有不可能状态设置为默认值 -100
        2. 注意，第一个 axis 是 to_label, 第二个 axis 是 from_label
        """
        self.transition.data[:, self.end] = -100.0      # end不可以向其他状态转移
        self.transition.data[self.start, :] = 100.0     # 任何状态不能转移到 start

        # 对 num_labels 两层遍历，排除所有不合理的情况
        for label, label_idx in self.label_vocab.items():
            if label.startswith("I-") or label.endswith("E-"):
                # <sos> 不能跳过 B 直接转移到 I 和 E
                self.transition.data[label_idx, self.start] = -100.0
            if label.startswith("B-") or label.endswith("I-"):
                # <eos> 不能由 B 或 I 转移得到 （BIOES规则）
                self.transition.data[self.end, label_idx] = 100.0

        for label_from, label_from_idx in self.label_vocab.items():
            if label_from=="O":
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-', 1)

            for label_to, label_to_idx in self.label_vocab.items():
                if label_to=='O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-', 1) # 1表示分割为俩部分

                if self.bioes:
                    # 1. 如果是 BIOES 形式，则
                    #   1) [O, E, S] 中的任意一个状态，都可以转移到
                    #      [O, B, S] 中任意一个状态，不论前后两个状态是否相同
                    #   ---- e.g., 可以从 E-PER 转移到 B-LOC
                    #   2) 当 label 相同时，允许 B->L, B->E, I->I, I->E
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    # 2. 如果是 BIO 形式，则
                    #   1) 任何一个状态都可能转移到 B 和 O
                    #   2) I 状态只能由相同 label 的 B 或者 I 得到
                    is_allowed = any(
                        [
                            label_to_prefix in ['B', 'O'],
                            label_from_prefix in ['B', 'O']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )

                if not is_allowed:
                    self.transition.data[label_to_idx, label_from_idx] = -100.0


    @staticmethod
    def pad_logits(logits: torch.Tensor)->torch.Tensor:
        """ 辅助 paddling 方法
        Padding the output of linear layer with <SOS> and <EOS> scores.
        经过这个变化，bert的logits就可以与CRF中的转移矩阵对应了
        :param logits: bert线性层的输出，logits
        :return:
        """
        batch_size, seq_len, _ = logits.size()      # (bz, seq_len, num_labels)
        pads = logits.new_full((batch_size, seq_len, 2),
                               -100.0,
                               requires_grad=False) # (bz, seq_len, 2)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    # 以下 5 个方法，用来在训练过程中计算得分
    def calc_binary_score(self, labels: torch.Tensor, lens: torch.Tensor)->torch.Tensor:
        """ 计算转移得分

        :param labels:  # (batch, seq_len)
        :param lens:    # (batch)
        :return:
        """
        batch_size, seq_len = labels.size()
        # 1. 扩展label：其实就是对labels在seq_len的维度上，扩展了一个开头和末尾，
        # 例如 [<BOS>, 我，是，好，人，<EOS>]
        # A tensor of size batch_size * (seq_len + 2),
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, -1, -1] = labels
        mask = sequence_mask(lens+1, max_len=(seq_len + 2)).long()
        pad_stop = labels.new_full((1, ), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        # 2. 扩展transition：复制了batch份，另batch中的每个instance都有一个transition矩阵
        trn = self.transition           # axis=0表示from，axis=1表示to
        trn_exp = trn.unsqueeze(0).expand(batch_size,
                                          self.label_size,
                                          self.label_size)

        # 3. 状态转移的计算（保存在to_label的得分中）
        lbl_r = labels[:, 1:]
        #       (batch, seq_len) -> (batch, seq_len+1, num_labels)
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.label_size)
        #       score of jumping to a tag
        #       取 trn_exp 的 lbl_rexp
        #       (batch, num_labels+2, num_labels+2) -> (batch, seq_len-1, num_labels+2)
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # 4. from_label 的得分计算
        lbl_lexp = labels[:, :, -1].unsqueeze(-1)           # (batch, seq_len+1, 1)
        trn_scr = torch.gather(trn_row, 1, lbl_lexp)        # (batch, seq_len+1, 1)
        trn_scr = trn_scr.squeeze(-1)                       # (batch, seq_len+1)

        # 5. mask 掉 seq_len 维度上的 start，注意不是 mask 掉 num_labels 上的 start
        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr
        return score


    def calc_unary_score(self, logits: torch.Tensor, labels: torch.Tensor, lens)->torch.Tensor:
        """ 计算发射得分，就是取出：每个位置上的正确状态（真实label）的概率值

        :param logits:  (batch, seq_len, num_label + 2)，概率分布
        :param labels:  (batch, seq_len)
        :param lens:    (batch)
        :return:        (batch, seq_len)
        """
        labels_exp = labels.unsqueeze(-1)
        # 根据维度 dim=2 索引，score[bz, seq_len, i] = labels_exp[i]
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores*mask
        return scores


    def calc_gold_score(self, logits, labels, lens):
        """ 计算真实得分

        :param logits:  (batch, seq_len, num_label + 2)
        :param labels:  (batch, seq_len)
        :param lens:    (batch)
        :return:    真实得分 = 发射得分 + 转移得分
        """
        unary_score = self.calc_unary_score(logits, labels, lens).sum(1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        """ 获取正确得分

        :param logits:
        :param lens:
        :return:
        """
        pass
        batch_size, _, _ = logits.size()
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0




    def loglik(self, logits: torch.Tensor, labels: torch.Tensor, lens):
        """ 计算损失

        :param logits:
        :param labels:
        :param lens:
        :return:
        """
        norm_score = self.calc_norm_score(logits, lens)
        gold_score = self.calc_gold_score(logits, labels, lens)
        return gold_score - norm_score

    # viterbi
    def viterbi_decode(self, logits, lens):
        """ pass

        :param logits:
        :param lens:
        :return:
        """


if __name__=="__main__":
    bert = AutoModel.from_pretrained("albert-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    num_hiddens = 768
    num_labels = 20

    text = "your text here."
    inputs = tokenizer(text, return_tensors='pt')
    bert_out = bert(inputs['input_ids'], attention_mask=inputs["attention_mask"])[0]
    print(bert_out.shape)                       # (batch size, seq_len, num_hiddens)

    # 用一个线性层将 num_hiddens 降维成 状态空间
    label_ffn = nn.Linear(num_hiddens, num_labels, bias=True)
    label_score = label_ffn(bert_out)           # (batch size, seq_len, vocab_size)
                                                # 每一个token对于状态的概率分布情况

    vocab = {}
    crf = CRF(vocab)
    label_score = crf.pad_logits(label_score)   # (batch size, seq_len, vocab_size+2)



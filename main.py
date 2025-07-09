import torch
import random
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json

from config import Config
from process import get_data
from model import PoetryModel, PoetryModel2
from utils import set_seed, set_logger

logger = logging.getLogger(__name__)


def split_train_test(data, train_ratio=0.8, shuffle=True):
    if shuffle:
        random.shuffle(data)
    total = len(data)
    train_total = int(total * train_ratio)
    train_data = data[:train_total]
    test_data = data[train_total:]
    print('总共有数据{}条'.format(total))
    print('划分后，训练集{}条'.format(train_total))
    print('划分后，测试集{}条'.format(total - train_total))
    return train_data, test_data


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        # 添加loss记录列表
        self.train_losses = []
        self.test_losses = []
        self.epochs = []
        # 添加混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def train(self, train_loader, test_loader=None):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        global_step = 0
        best_test_loss = float("inf")
        best_epoch = None
        total_step = len(train_loader) * self.config.num_epoch
        
        for epoch in range(1, self.config.num_epoch + 1):
            total_loss = 0.
            for train_step, train_data in enumerate(train_loader):
                self.model.train()
                train_data = train_data.long().to(self.config.device, non_blocking=True)
                input = train_data[:, :-1]
                target = train_data[:, 1:]
                
                optimizer.zero_grad()
                
                # 使用混合精度训练
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output, _ = self.model(input)
                        active = (input > 0).view(-1)
                        active_output = output[active]
                        active_target = target.contiguous().view(-1)[active]
                        loss = self.criterion(active_output, active_target)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output, _ = self.model(input)
                    active = (input > 0).view(-1)
                    active_output = output[active]
                    active_target = target.contiguous().view(-1)[active]
                    loss = self.criterion(active_output, active_target)
                    loss.backward()
                    optimizer.step()
                
                total_loss = total_loss + loss.item()
                
                if global_step % 100 == 0:  # 减少日志输出频率
                    logger.info('epoch:{} step:{}/{} loss:{}'.format(
                        epoch, global_step, total_step, loss.item()
                    ))
                global_step += 1
            
            # 记录每个epoch的平均训练loss
            avg_train_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            self.epochs.append(epoch)
            
            logger.info('epoch:{} avg_train_loss:{}'.format(
                epoch, avg_train_loss
            ))
            
            if self.config.do_test:
                test_loss = self.test(test_loader)
                # 记录测试loss
                avg_test_loss = test_loss / len(test_loader)
                self.test_losses.append(avg_test_loss)
                
                if test_loss < best_test_loss:
                    torch.save(self.model.state_dict(), self.config.save_path)
                    best_test_loss = test_loss
                    best_epoch = epoch
                logger.info('epoch:{} avg_test_loss:{}'.format(epoch, avg_test_loss))
        
        logger.info('====================')
        logger.info('在第{}个epoch损失最小为：{}'.format(best_epoch, best_test_loss))
        
        # 训练结束后绘制并保存loss图
        self.plot_and_save_loss()

    def test(self, test_loader):
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            for test_step, test_data in enumerate(test_loader):
                test_data = test_data.long().to(self.config.device)
                input = test_data[:, :-1]
                target = test_data[:, 1:]
                output, _ = self.model(input)
                active = (input > 0).view(-1)
                active_output = output[active]
                active_target = target.contiguous().view(-1)[active]
                loss = self.criterion(active_output, active_target)
                total_loss = total_loss + loss.item()
        return total_loss

    def generate(self, start_words, prefix_words=None):
        """
        给定几个词，根据这几个词接着生成一首完整的诗歌
        start_words：u'春江潮水连海平'
        比如start_words 为 春江潮水连海平，可以生成：

        """

        results = list(start_words)
        start_word_len = len(start_words)
        # 手动设置第一个词为<SOP>
        input = torch.tensor([self.config.word2idx['SOP']]).view(1, 1).long()
        input = input.to(self.config.device)
        hidden = None

        if prefix_words:
            for word in prefix_words:
                output, hidden = self.model(input, hidden)
                input = input.data.new([self.config.word2idx[word]]).view(1, 1)

        for i in range(self.config.max_gen_len):
            # 初始化的时候input=[[2]], hidden=None
            output, hidden = self.model(input, hidden)

            if i < start_word_len:
                w = results[i]
                input = input.data.new([self.config.word2idx[w]]).view(1, 1)
            else:
                top_index = output.data[0].topk(1)[1][0].item()
                w = self.config.idx2word[top_index]
                results.append(w)
                input = input.data.new([top_index]).view(1, 1)
            if w == 'EOP':
                del results[-1]
                break
        return results

    def gen_acrostic(self, start_words, prefix_words=None):
        """
        生成藏头诗
        start_words : u'深度学习'
        生成：
        深木通中岳，青苔半日脂。
        度山分地险，逆浪到南巴。
        学道兵犹毒，当时燕不移。
        习根通古岸，开镜出清羸。
        """
        results = []
        start_word_len = len(start_words)
        input = (torch.tensor([self.config.word2idx['SOP']]).view(1, 1).long())
        input = input.to(self.config.device)
        hidden = None

        index = 0  # 用来指示已经生成了多少句藏头诗
        # 上一个词
        pre_word = 'SOP'

        if prefix_words:
            for word in prefix_words:
                output, hidden = self.model(input, hidden)
                input = (input.data.new([self.config.word2idx[word]])).view(1, 1)

        for i in range(self.config.max_gen_len):
            output, hidden = self.model(input, hidden)
            top_index = output.data[0].topk(1)[1][0].item()
            w = self.config.idx2word[top_index]

            if (pre_word in {u'。', u'！', 'SOP'}):
                # 如果遇到句号，藏头的词送进去生成

                if index == start_word_len:
                    # 如果生成的诗歌已经包含全部藏头的词，则结束
                    break
                else:
                    # 把藏头的词作为输入送入模型
                    w = start_words[index]
                    index += 1
                    input = (input.data.new([self.config.word2idx[w]])).view(1, 1)
            else:
                # 否则的话，把上一次预测是词作为下一个词输入
                input = (input.data.new([self.config.word2idx[w]])).view(1, 1)
            results.append(w)
            pre_word = w
        return results

    def plot_and_save_loss(self):
        """
        绘制并保存训练和测试的loss曲线
        """
        plt.figure(figsize=(12, 5))
        
        # 绘制训练loss
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 如果有测试loss，也绘制测试loss
        if self.test_losses:
            plt.subplot(1, 2, 2)
            plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
            plt.plot(self.epochs, self.test_losses, 'r-', label='Test Loss', linewidth=2)
            plt.title('Training vs Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 确保保存目录存在
        save_dir = './results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'loss_curve.pdf'), bbox_inches='tight')
        
        print(f"Loss曲线已保存到 {save_dir}/loss_curve.png 和 {save_dir}/loss_curve.pdf")
        
        # 显示图片（可选，如果在jupyter notebook中运行）
        plt.show()
        
        # 保存loss数据到文件
        loss_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses if self.test_losses else []
        }
        
        with open(os.path.join(save_dir, 'loss_data.json'), 'w', encoding='utf-8') as f:
            json.dump(loss_data, f, indent=2, ensure_ascii=False)
        
        print(f"Loss数据已保存到 {save_dir}/loss_data.json")


if __name__ == '__main__':
    config = Config()
    set_seed(123)
    set_logger('./main.log')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    
    # 添加GPU信息打印
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("未检测到CUDA GPU，使用CPU训练")

    data, word2idx, idx2word = get_data(config)
    config.word2idx = word2idx
    config.idx2word = idx2word
    train_data, test_data = split_train_test(data)

    if config.do_train:
        train_data = torch.from_numpy(train_data)
        train_loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows上设为0可能更快
            pin_memory=True,  # 加速GPU传输
        )

    if config.do_test:
        test_data = torch.from_numpy(test_data)
        test_loader = DataLoader(
            test_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,  # Windows上设为0可能更快
            pin_memory=True,  # 加速GPU传输
        )

    model = PoetryModel2(len(word2idx), config.embedding_dim, config.hidden_dim)
    if config.do_load_model:
        print('加载已训练好的模型。。。')
        model.load_state_dict(torch.load(config.load_path))

    model.to(device)

    trainer = Trainer(model, config)
    if config.do_train:
        if config.do_test:
            trainer.train(train_loader, test_loader)
        else:
            trainer.train(train_loader)

    if config.do_predict:
        result = trainer.generate('丽日照残春')
        print("".join(result))
        result = trainer.gen_acrostic('深度学习')
        print("".join(result))

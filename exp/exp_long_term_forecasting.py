from matplotlib import pyplot as plt

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # 历史数据
                    hist = input[0, :, -1]  # input: [B, L, N]
                    # 真实未来
                    true_f = true[0, :, -1]
                    # 预测未来
                    pred_f = pred[0, :, -1]
                    # 调用新版 visual
                    visual(hist, true_f, pred_f, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    #
    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #
    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, :]
    #             batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()
    #             if test_data.scale and self.args.inverse:
    #                 shape = batch_y.shape
    #                 if outputs.shape[-1] != batch_y.shape[-1]:
    #                     outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
    #                 outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
    #                 batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
    #
    #             outputs = outputs[:, :, f_dim:]
    #             batch_y = batch_y[:, :, f_dim:]
    #
    #             pred = outputs
    #             true = batch_y
    #
    #             preds.append(pred)
    #             trues.append(true)
    #             if i % 20 == 0:
    #                 input = batch_x.detach().cpu().numpy()
    #                 if test_data.scale and self.args.inverse:
    #                     shape = input.shape
    #                     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
    #                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                 # （替换绘图块）—— 横向长方形，无网格，图例灰色背景
    #
    #                 total_len = gt.shape[0]  # 通常为 hist + pred (e.g. 96 + 96 = 192)
    #
    #                 # 画布：横向长方形（横比竖稍长）
    #                 fig, ax = plt.subplots(figsize=(7, 4))
    #
    #                 # 画线（细线）
    #                 ax.plot(np.arange(total_len), gt, color='green', linewidth=1.0, solid_capstyle='round')
    #                 ax.plot(np.arange(total_len), pd, color='red', linewidth=1.0, solid_capstyle='round')
    #
    #                 # 分界线（历史/预测），如不需要可删掉下一行
    #                 ax.axvline(x=input.shape[1] - 0.5, color='gray', linestyle='--', linewidth=0.8)
    #
    #                 # 横轴自适应长度：左右各留一点空白（不强制到200）
    #                 left_pad = 4
    #                 right_pad = 4
    #                 ax.set_xlim(-left_pad, total_len - 1 + right_pad)
    #
    #                 # 刻度：尽量以25为步长（若长度不足则退化）
    #                 if total_len >= 25:
    #                     max_tick = (total_len // 25) * 25
    #                     if max_tick < total_len:
    #                         max_tick = min(total_len, max_tick + 25)
    #                     ticks = np.arange(0, max_tick + 1, 25)
    #                 else:
    #                     step = max(1, total_len // 5)
    #                     ticks = np.arange(0, total_len + 1, step)
    #                 ax.set_xticks(ticks)
    #
    #                 # 去掉网格
    #                 ax.grid(False)
    #
    #                 # 图例：灰色长方形框
    #                 leg = ax.legend(['Ground Truth', 'Prediction'], loc='upper right', fontsize=9, frameon=True)
    #                 frame = leg.get_frame()
    #                 frame.set_facecolor('#e0e0e0')  # 浅灰背景
    #                 frame.set_edgecolor('#bdbdbd')  # 灰边框
    #                 frame.set_linewidth(0.6)
    #
    #                 # 若不想显示坐标轴标签，可以注释掉下面两行
    #                 ax.set_xlabel('Time step', fontsize=10)
    #                 ax.set_ylabel('Value', fontsize=10)
    #
    #                 # 微调边框：去掉上/右边框（保留下/左）
    #                 ax.spines['top'].set_visible(False)
    #                 ax.spines['right'].set_visible(False)
    #
    #                 # 保存（保持高分辨率）
    #                 os.makedirs(folder_path, exist_ok=True)
    #                 plt.tight_layout()
    #                 print(111)
    #                 plt.savefig(os.path.join(folder_path, f"{i}.pdf"), bbox_inches='tight', dpi=300)
    #                 plt.close(fig)
    #
    #     preds = np.concatenate(preds, axis=0)
    #     trues = np.concatenate(trues, axis=0)
    #     print('test shape:', preds.shape, trues.shape)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)
    #
    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     # dtw calculation
    #     if self.args.use_dtw:
    #         dtw_list = []
    #         manhattan_distance = lambda x, y: np.abs(x - y)
    #         for i in range(preds.shape[0]):
    #             x = preds[i].reshape(-1, 1)
    #             y = trues[i].reshape(-1, 1)
    #             if i % 100 == 0:
    #                 print("calculating dtw iter:", i)
    #             d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
    #             dtw_list.append(d)
    #         dtw = np.array(dtw_list).mean()
    #     else:
    #         dtw = 'Not calculated'
    #
    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
    #     f = open("result_long_term_forecast.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()
    #
    #     np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(folder_path + 'pred.npy', preds)
    #     np.save(folder_path + 'true.npy', trues)
    #
    #     return

    # def test(self, setting, test=0):
    #     # --- 必要 imports 必须在使用 os 之前 ---
    #     import os
    #     import numpy as np
    #     import matplotlib
    #     matplotlib.use('Agg')  # 如果在无显示器的服务器上运行
    #     import matplotlib.pyplot as plt
    #
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         # 此处能安全使用 os
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    #
    #     self.model.eval()
    #
    #     # 准备容器
    #     hists = []  # 保存历史输入（inverse 后）
    #     preds = []
    #     trues = []
    #     mse_list = []  # 每个样本的 mse（对时间和通道平均）
    #
    #     vis_folder = './test_results/' + setting + '/'
    #     if not os.path.exists(vis_folder):
    #         os.makedirs(vis_folder)
    #
    #     # 全局样本索引计数器（从0开始）
    #     global_sample_idx = 0
    #
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #
    #             # forward
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #
    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, :]
    #             batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()
    #
    #             # inverse scale outputs / batch_y if needed
    #             if test_data.scale and self.args.inverse:
    #                 shape_out = batch_y.shape
    #                 if outputs.shape[-1] != batch_y.shape[-1]:
    #                     outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
    #                 outputs = test_data.inverse_transform(outputs.reshape(shape_out[0] * shape_out[1], -1)).reshape(
    #                     shape_out)
    #                 batch_y = test_data.inverse_transform(batch_y.reshape(shape_out[0] * shape_out[1], -1)).reshape(
    #                     shape_out)
    #
    #             # inverse for batch_x (history) for saving/plotting
    #             input_np = batch_x.detach().cpu().numpy()
    #             if test_data.scale and self.args.inverse:
    #                 shape_in = input_np.shape
    #                 input_np = test_data.inverse_transform(input_np.reshape(shape_in[0] * shape_in[1], -1)).reshape(
    #                     shape_in)
    #
    #             # crop feature dimension according to f_dim
    #             outputs = outputs[:, :, f_dim:]
    #             batch_y = batch_y[:, :, f_dim:]
    #             input_np = input_np[:, :, f_dim:]
    #
    #             # append to lists
    #             batch_pred = outputs
    #             batch_true = batch_y
    #             preds.append(batch_pred)
    #             trues.append(batch_true)
    #             hists.append(input_np)
    #
    #             # compute per-sample mse (average over time and channels)
    #             batch_mse = ((batch_pred - batch_true) ** 2).mean(axis=(1, 2))  # shape (B,)
    #             mse_list.append(batch_mse)
    #
    #             # visualization snapshot: 每隔 20 个 batch 保存该 batch 第 0 个样本的图
    #             if i % 20 == 0:
    #                 s = 0
    #                 global_idx = global_sample_idx + s
    #
    #                 # pick last channel (-1) for plotting as before
    #                 hist_arr = input_np[s, :, -1]
    #                 true_fut = batch_true[s, :, -1]
    #                 pred_fut = batch_pred[s, :, -1]
    #                 gt = np.concatenate((hist_arr, true_fut), axis=0)
    #                 pd = np.concatenate((hist_arr, pred_fut), axis=0)
    #
    #                 sample_mse = float(batch_mse[s])
    #
    #                 # ===== 绘图开始（按你的要求） =====
    #                 total_len = gt.shape[0]  # e.g., 96 + 96 = 192
    #                 fig, ax = plt.subplots(figsize=(6, 4))
    #
    #                 # GT: green, Pred: red, 线条细
    #                 ax.plot(np.arange(total_len), pd, color='green', linewidth=1.2, solid_capstyle='round',
    #                         label='Ground Truth')
    #                 # 红色 = 仅预测部分
    #                 hist_len = input.shape[1]
    #                 ax.plot(np.arange(hist_len, total_len), pd[hist_len:],
    #                         color='red', linewidth=1.0, solid_capstyle='round', label='Prediction')
    #
    #                 # history / prediction 分界（你想去掉也可以直接删除这行）
    #                 # ax.axvline(x=hist_arr.shape[0] - 0.5, color='gray', linestyle='--', linewidth=0.8)
    #
    #                 # 横轴固定 0..200，刻度每 25，左侧留白
    #                 ax.set_xlim(-4, 200)
    #                 ax.set_xticks(np.arange(0, 201, 25))
    #
    #                 # 去掉网格（按你要求）
    #                 # ax.grid(False)  # 网格不显示
    #
    #                 # 添加 legend（颜色标注）
    #                 # ax.legend(loc='upper right', fontsize=10, frameon=False)
    #                 #
    #                 # # 若不想显示 xlabel/ylabel，可注释下面两行
    #                 # ax.set_xlabel('Time step', fontsize=10)
    #                 # ax.set_ylabel('Value', fontsize=10)
    #
    #                 # 微调边框样式
    #                 # ax.spines['top'].set_visible(False)
    #                 # ax.spines['right'].set_visible(False)
    #
    #                 # 保存
    #                 pdf_name = f"{i}_sample{global_idx}_mse{sample_mse:.6f}.pdf"
    #                 os.makedirs(vis_folder, exist_ok=True)
    #                 plt.tight_layout()
    #                 plt.savefig(os.path.join(vis_folder, pdf_name), bbox_inches='tight', dpi=300)
    #                 plt.close(fig)
    #                 # ===== 绘图结束 =====
    #
    #             # update global idx
    #             global_sample_idx += batch_pred.shape[0]
    #
    #     # end with torch.no_grad()
    #
    #     # concat and save results
    #     preds = np.concatenate(preds, axis=0)
    #     trues = np.concatenate(trues, axis=0)
    #     hists = np.concatenate(hists, axis=0)
    #     mse_per_sample = np.concatenate(mse_list, axis=0)
    #
    #     print('test shape:', preds.shape, trues.shape, hists.shape)
    #
    #     results_folder = './results/' + setting + '/'
    #     if not os.path.exists(results_folder):
    #         os.makedirs(results_folder)
    #
    #     np.save(results_folder + 'hist.npy', hists)
    #     np.save(results_folder + 'mse_per_sample.npy', mse_per_sample)
    #     np.save(results_folder + 'pred.npy', preds)
    #     np.save(results_folder + 'true.npy', trues)
    #
    #     return


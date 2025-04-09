
import torch
import os
import time
from tensorboardX import SummaryWriter
import numpy as np
from agent1 import DQNAgent
from agent1 import DQN
from env1 import envCube
from train import index, Loader, get_metrics, get_metrics1, train_test, train_test1
from sklearn import metrics
from datapro import CVEdgeDataset
from model import HRRLMDA, EmbeddingM, EmbeddingD,MDI
from datetime import datetime
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as DataLoader
from sklearn.utils import shuffle
from matplotlib import pyplot
class DDQN:
    def __init__(self):
        self.path = os.path.realpath(__file__)
        self.filename = os.path.splitext(os.path.basename(self.path))[0]
        self.writer = SummaryWriter(f'logs/{self.filename}')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episodes = 10
        self.replay_memory_size = 60
        self.batch_size = 10
        self.discount = 0.85
        self.learning_rate = 0.001
        self.update_target_mode_every = 50
        self.statistics_every = 50
        self.model_save_avg_reward = 90
        self.epi_start = 1
        self.epi_end = 0.001
        self.epi_decay = 0.999995
        self.visualize = 0
        self.verbose = 1
        self.show_every = 10
        self.kfold = 5
    def train(self,train_dex, valid_dex, sample, true_label, simData, train_data, param, allneg_samples):
        env = envCube()
        agent = DQNAgent(env.OBSERVATION_SPACE_VALUES, env.ACTION_SPACE_VALUES, self.replay_memory_size, self.batch_size, self.discount,self.learning_rate , self.epi_start, self.epi_end, self.epi_decay,self.device)
        train_edges, train_labels, test_edges, test_labels, train_idx, valid_idx = index(train_data, param, state='valid')
        test_metric = []
        auc_result = []
        prc_result = []
        fprs = []
        tprs = []
        recalls = []
        precisions = []
        score_0 = []
        for i in range(5):
            train_edges, test_edges = sample[train_dex[i]], sample[valid_dex[i]]
            train_labels, test_labels = true_label[train_dex[i]], true_label[valid_dex[i]]
            train_edges, train_labels = shuffle(train_edges, train_labels, random_state=42)
            test_edges, test_labels = shuffle(test_edges, test_labels, random_state=42)
            b = i + 1
            model = Loader(i, simData, train_data, param, train_edges, train_labels, train_idx, valid_idx, state='valid')
            trainEdges = CVEdgeDataset(train_edges, train_labels)
            trainLoader1 = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=False, num_workers=0)
            item1 = []
            for i, item in enumerate(trainLoader1):
                item1.append([item])
            for e in range(param.epoch):
                c = 0
                pre_score, state, loss, c = env.reset(c, item1, simData, train_data, param, trainLoader1, test_edges, test_labels, model)                                 #重置环境
                done = False
                episode_reward = 0                                  # 每局奖励清零
                a = 0
                print("-----training-----")
                for i in range(len(item1)-1):
                    a = a + 1
                    action = agent.select_action(state)             # 选择action
                    param = env.step1(action, param)
                    # param = env.step2(action, param)
                    next_state, reward, loss, c = env.step(a, c, item1, simData, train_data, param, trainLoader1, test_edges, test_labels, model, state, action, loss)     #游戏走一步
                    agent.push_transition(state, action, reward, next_state, done)   # 将当前状态放入池
                    agent.update_epsilon()                          # 更新epsilon
                    agent.update_model()                            # 更新model
                    state = next_state                              # 更新state
                    episode_reward += reward                        # 累加当次训练的reward
                    agent.episode_rewards.append(episode_reward)  # 收集所有训练累计的reward
                    print(f"After batch {i + 1}: loss= {loss:.3f};", end='\n')
                    if loss < 0.1000:
                        break
            valid_score, valid_label = [], []
            valid_metric = []
            model.eval()
            testEdges = CVEdgeDataset(test_edges, test_labels)
            testLoader = DataLoader.DataLoader(testEdges, batch_size=param.batchSize, shuffle=False, num_workers=0)
            print("-----validing-----")
            item2 = []
            for i, item in enumerate(testLoader):
                item2.append([item])
            c = 0
            for i in range(len(item2)):
                data, label = item2[c][0]
                pre_score, result, loss, c = train_test(c, train_data, item2, simData, param, trainLoader1, test_edges, test_labels, model)
                print(f"After batch {i + 1}: loss= {loss:.3f};", end='\n')
                batch_score = pre_score.cpu().detach().numpy()
                valid_score = np.append(valid_score, batch_score)
                valid_label = np.append(valid_label, label.numpy())
            metric = get_metrics(valid_score, valid_label)
            valid_metric.append(metric)
            metrics, test_auc, test_prc, fpr, tpr, precision_u, recall_u = get_metrics1(valid_score, valid_label)
            prc_result.append(test_prc)
            auc_result.append(test_auc)
            fprs.append(fpr)
            tprs.append(tpr)
            recalls.append(recall_u)
            precisions.append(precision_u)
            test_metric.append(metrics)
            now = datetime.now()
            end = now.strftime
        #实验结果展示
        mean_recall = np.linspace(0, 1, 10000)
        precision = []
        for i in range(len(recalls)):
            precision.append(interp(1 - mean_recall, 1 - recalls[i], precisions[i]))
            precision[-1][0] = 1.0
            plt.plot(recalls[i], precisions[i], alpha=0.4, label='PR Fold %d (AUPR = %0.4f)' % (i + 1, prc_result[i][0]))

        mean_precision = np.mean(precision, axis=0)
        mean_precision[-1] = 0
        mean_prc = np.mean(prc_result)
        prc_std = np.std(prc_result)
        # np.savetxt("mean_recall203.txt", mean_recall, fmt="%.5f", delimiter=",")
        # np.savetxt("mean_precision203.txt", mean_precision, fmt="%.5f", delimiter=",")
        plt.plot(mean_recall, mean_precision, '--', linewidth=1.5, color='g', label='Mean PR (AUPR = %0.4f)' % mean_prc)  # AP: Average Precision
        # plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('P-R curves')
        plt.legend(loc='lower left')
        plt.show()

        mean_fpr = np.linspace(0, 1, 10000)
        tpr = []
        for i in range(len(fprs)):
            tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
            tpr[-1][0] = 0.0
            plt.plot(fprs[i], tprs[i], alpha=0.4, label='ROC fold %d (AUC = %0.4f)' % (i + 1, auc_result[i][0]))

        mean_tpr = np.mean(tpr, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(auc_result)
        auc_result_std = np.std(auc_result)
        # np.savetxt("mean_fpr203.txt", mean_fpr, fmt="%.5f", delimiter=",")
        # np.savetxt("mean_tpr203.txt", mean_tpr, fmt="%.5f", delimiter=",")
        pyplot.plot(mean_fpr, mean_tpr, '--', linewidth=1.5, color='r', label='Mean ROC (AUC = %0.4f)' % mean_auc)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('ROC Curves')
        plt.legend(loc='lower left')
        pyplot.show()
        miRNAnumber = np.genfromtxt(r'miR_name.txt', dtype=str, delimiter='\t')
        diseasenumber = np.genfromtxt(r'dis_name.txt', dtype=str, delimiter='\t')
        torch.save(model, "./savemodel/fold_{}.pkl".format(b))
        model = torch.load("./savemodel/fold_{}.pkl".format(b))
        model.eval()
        batch_size = 1000
        def batch_generator(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield data[i:i + batch_size]
        for batch in batch_generator(allneg_samples, batch_size):
            zuizhongscore, node_embed, total_loss = train_test1(batch, train_data, param, simData, model)
            del node_embed, total_loss
            zuizhongscore = zuizhongscore.ravel()
            score_0.extend(zuizhongscore)
        score_0 = np.loadtxt("zuizhongscore_results.txt", delimiter=",", skiprows=1)
        score_0 = score_0[:, 1]
        score_0_sorted = sorted(score_0, reverse=True)
        score_00 = np.array(score_0)
        score_0ranknumber = np.transpose(np.argsort(-score_00))

        diseaserankname_pos = allneg_samples[score_0ranknumber, 1]
        diseaserankname = diseasenumber[diseaserankname_pos, 1]
        diseaserankname = diseaserankname.reshape(491677, )

        miRNArankname_pos = allneg_samples[score_0ranknumber, 0]
        miRNArankname = miRNAnumber[miRNArankname_pos, 1]
        miRNArankname = miRNArankname.reshape(491677, )

        score_0rank_pd = pd.Series(score_0_sorted)
        diseaserankname_pd = pd.Series(diseaserankname)
        miRNArankname_pd = pd.Series(miRNArankname)
        prediction_0_out = pd.concat([diseaserankname_pd, miRNArankname_pd, score_0rank_pd], axis=1)
        prediction_0_out.columns = ['Disease', 'miRNA', 'Score']
        prediction_0_out.to_excel(r'prediction result.xlsx', sheet_name='Sheet1', index=False)
        print('---' * 1)
import os
import sys
import time
import torch
import numpy as np
from math import isnan

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class Runner(object):
    def __init__(self):
        self.start_train = 0
        self.load_feature()
        self.model = load_model(args).to(args.device)
        self.loss = ReconLoss(args) if args.model not in ['DynVAE', 'VGRNN', 'HVGRNN'] else VGAEloss(args)

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
            logger.info("using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                if args.dataset == 'disease':
                    feature = sp.load_npz(disease_path).toarray()
                self.x = torch.from_numpy(feature).float().to(args.device)
                logger.info('using pre-defined feature')
            else:
                self.x = torch.eye(args.num_nodes).to(args.device)
                logger.info('using one-hot feature')
            args.nfeat = self.x.size(1)

    def optimizer(self, using_riemannianAdam=True):
        if using_riemannianAdam:
            import geoopt
            optimizer = geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr,
                                                          weight_decay=args.weight_decay)
        else:
            import torch.optim as optim
            optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def run(self):
        optimizer = self.optimizer()
        t_total0 = time.time()
        test_results, min_loss = [0] * 5, 10
        self.model.train()
        for epoch in range(1, args.max_epoch + 1):
            t0 = time.time()
            epoch_losses = []
            self.model.init_hiddens()
            self.model.train()
            t = 0
            edge_index, pos_index, neg_index, adj, activate_nodes, edge_weight = prepare(data, t)
            optimizer.zero_grad()
            all = edge_index, adj
            z, c = self.model(all, self.x)
            if args.use_htc == 0:
                epoch_loss = self.loss(z, edge_index)
            else:
                epoch_loss = self.loss(z, edge_index, c) + self.model.htc(z, c)
            epoch_loss.backward()
            optimizer.step()
            epoch_losses.append(epoch_loss.item())
            self.model.update_hiddens_all_with(z)
            self.model.eval()
            average_epoch_loss = np.mean(epoch_losses)
            if average_epoch_loss < min_loss:
                min_loss = average_epoch_loss
                test_results = self.test(epoch, c, z)
                patience = 0
            else:
                patience += 1
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info('==' * 27)
                logger.info("Epoch:{}, Loss: {:.4f}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, average_epoch_loss,
                                                                                          time.time() - t0,
                                                                                          gpu_mem_alloc))
                logger.info(
                    "Epoch:{:}, Test AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, F1: {:.4f}".
                        format(test_results[0], test_results[1], test_results[2], test_results[3], test_results[4]))
            if isnan(epoch_loss):
                print('nan loss')
                break
        logger.info('>> Total time : %6.2f' % (time.time() - t_total0))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

    def test(self, epoch, c, embeddings=None):
        auc_test_list, ap_test_list, acc_test_list, reall_test_list, pre_test_list, f1_test_list, roc_test_list = [], [], [], [], [], [], []
        embeddings = embeddings.detach()
        # for t in self.test_shots:
        t = 0
        edge_index, pos_edge, neg_edge = prepare(data, t)[:3]
        auc_test, ap_test, acc, f1 = self.loss.predict(embeddings, pos_edge, neg_edge, t, c)
        auc_test_list.append(auc_test)
        ap_test_list.append(ap_test)
        acc_test_list.append(acc)
        f1_test_list.append(f1)
        return epoch, np.mean(auc_test_list), np.mean(ap_test_list), np.mean(acc_test_list), np.mean(f1_test_list)



if __name__ == '__main__':
    from script.config import args
    from script.utils.util import set_random, logger, init_logger, disease_path
    from script.models.load_model import load_model
    from script.loss import ReconLoss, VGAEloss
    from script.utils.data_util import loader, prepare_dir
    from script.inits import prepare

    data = loader(dataset=args.dataset)
    args.num_nodes = data['num_nodes']
    set_random(args.seed)
    init_logger(prepare_dir(args.output_folder) + args.dataset + '.txt')
    runner = Runner()
    runner.run()

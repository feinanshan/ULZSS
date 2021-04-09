import numpy as np
import pdb

class runningScore(object):
    def __init__(self, n_classes,unseen_cls):
        self.n_classes = n_classes

        self.all_cls = range(n_classes)
        self.unseen_cls = unseen_cls
        self.seen_cls = [i for i in self.all_cls if i not in self.unseen_cls]
        self.confusion_matrix = np.zeros((n_classes, 3)) #M,I,U

    def update(self, seg_acc):
        for (cat,M,I,U) in seg_acc:
            self.confusion_matrix[cat][0] = self.confusion_matrix[cat][0] + M
            self.confusion_matrix[cat][1] = self.confusion_matrix[cat][1] + I
            self.confusion_matrix[cat][2] = self.confusion_matrix[cat][2] + U

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix


        acc_all = hist[self.all_cls,1].sum()/hist[self.all_cls,0].sum()
        acc_unseen = hist[self.unseen_cls,1].sum()/hist[self.unseen_cls,0].sum()
        acc_seen = hist[self.seen_cls,1].sum()/hist[self.seen_cls,0].sum()

        macc_all = np.nanmean(hist[self.all_cls,1]/hist[self.all_cls,0])
        macc_unseen = np.nanmean(hist[self.unseen_cls,1]/hist[self.unseen_cls,0])
        macc_seen = np.nanmean(hist[self.seen_cls,1]/hist[self.seen_cls,0])

        miou_all = np.nanmean(hist[self.all_cls,1]/hist[self.all_cls,2])
        miou_unseen = np.nanmean(hist[self.unseen_cls,1]/hist[self.unseen_cls,2])
        miou_seen = np.nanmean(hist[self.seen_cls,1]/hist[self.seen_cls,2])

        acc_ =  '%.5f | %.5f | %.5f'%(acc_all,acc_unseen,acc_seen)
        macc_ = '%.5f | %.5f | %.5f'%(macc_all,macc_unseen,macc_seen)
        miou_ = '%.5f | %.5f | %.5f'%(miou_all,miou_unseen,miou_seen)

        pdb.set_trace()

        return {
                "Overall Acc: |all|unseen|seen|\t": acc_,
                "Mean Acc : |all|unseen|seen|\t": macc_,
                "Mean IoU : |all|unseen|seen|\t": miou_,
            },miou_all

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes,3))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

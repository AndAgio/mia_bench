from typing import Any


class EpochStats():
    def __init__(self):
        self.train_loss: float = 0
        self.train_correct: float = 0
        self.train_total: float = 0
        self.test_loss: float = 0
        self.test_correct: float = 0
        self.test_total: float = 0

    def increase(self, field: str, by: float):
        if field == 'train_loss':
            self.train_loss += by
        elif field == 'train_correct':
            self.train_correct += by
        elif field == 'train_total':
            self.train_total += by
        elif field == 'test_loss':
            self.test_loss += by
        elif field == 'test_correct':
            self.test_correct += by
        elif field == 'test_total':
            self.test_total += by
        else:
            raise ValueError('Field {} not found!'.format(field))

    def avg_loss(self, mode: str):
        assert mode in ['train', 'test']
        if mode == 'train':
            return self.train_loss / self.train_total
        else:
            return self.test_loss / self.test_total
        
    def avg_acc(self, mode: str):
        assert mode in ['train', 'test']
        if mode == 'train':
            return self.train_correct / self.train_total
        else:
            return self.test_correct / self.test_total


class TrainStats():
    def __init__(self, 
                epoch: int = 1,
                best_train_acc: float = 0,
                best_test_acc: float = 0,
                elapsed_time: float = 0,
                train_accs: list[float] = [],
                test_accs: list[float] = []):
        self.epoch = epoch
        self.best_train_acc = best_train_acc
        self.best_test_acc = best_test_acc
        self.elapsed_time = elapsed_time
        self.train_accs = train_accs
        self.test_accs = test_accs

    def reset(self):
        self.epoch: int = 1
        self.best_train_acc: float = 0
        self.best_test_acc: float = 0
        self.elapsed_time: float = 0
        self.train_accs: list[float] = []
        self.test_accs: list[float] = []

    def increase_epoch(self, by: int = 1):
        self.epoch += by
    
    def update_train_accs(self, acc):
        self.train_accs.append(acc)
        if acc > self.best_train_acc:
            self.best_train_acc = acc

    def update_test_accs(self, acc):
        self.test_accs.append(acc)
        if acc > self.best_test_acc:
            self.best_test_acc = acc

    def update_time(self, by: float):
        self.elapsed_time += by
    
    def is_new_best(self, acc: float, mode: str):
        assert mode in ['train', 'test']
        if mode == 'train':
            if acc > self.best_train_acc:
                return True
        else:
            if acc > self.best_test_acc:
                return True
        return False


class TrainCheckpoint(TrainStats):
    def __init__(self, train_stats: TrainStats, model_state: dict[str, Any] = None, opt_state: dict[str, Any] = None, sched_state: dict[str, Any] = None):
        super().__init__(train_stats.epoch, train_stats.best_train_acc, train_stats.best_test_acc, train_stats.elapsed_time, train_stats.train_accs, train_stats.test_accs)
        self.model_state: dict[str, Any] = model_state
        self.optimizer_state: dict[str, Any] = opt_state
        self.scheduler_state: dict[str, Any] = sched_state

    def update(self, train_stats: TrainStats, model_state: dict[str, Any], opt_state: dict[str, Any], sched_state: dict[str, Any]):
        self.epoch = train_stats.epoch
        self.best_train_acc = train_stats.best_train_acc
        self.best_test_acc = train_stats.best_test_acc
        self.elapsed_time = train_stats.elapsed_time, 
        self.train_accs = train_stats.train_accs
        self.test_accs = train_stats.test_accs
        self.model_state = model_state
        self.optimizer_state = opt_state
        self.scheduler_state = sched_state
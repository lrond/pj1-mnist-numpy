import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.train_steps = []
        self.dev_steps = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        eval_interval = kwargs.get("eval_interval", log_iters)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]
            num_batches = int(np.ceil(X.shape[0] / self.batch_size))

            for iteration in range(num_batches):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]
                if train_X.shape[0] == 0:
                    continue

                global_step = epoch * num_batches + iteration
                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                self.train_steps.append(global_step)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                should_eval = (
                    eval_interval is not None
                    and eval_interval > 0
                    and ((iteration) % eval_interval == 0 or iteration == num_batches - 1)
                )
                if should_eval:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                    self.dev_steps.append(global_step)
                    if dev_score > best_score:
                        save_path = os.path.join(save_dir, 'best_model.pickle')
                        self.save_model(save_path)
                        print(f"best accuracy performance has been updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score
                else:
                    dev_score = self.dev_scores[-1] if self.dev_scores else None
                    dev_loss = self.dev_loss[-1] if self.dev_loss else None

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    if dev_score is not None:
                        print(f"[Dev] loss: {dev_loss}, score: {dev_score}")
        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits_list = []
        loss_sum = 0.0
        total = 0
        for start in range(0, X.shape[0], self.batch_size):
            batch_X = X[start:start + self.batch_size]
            batch_y = y[start:start + self.batch_size]
            if batch_X.shape[0] == 0:
                continue
            batch_logits = self.model(batch_X)
            batch_loss = self.loss_fn(batch_logits, batch_y)
            logits_list.append(batch_logits)
            loss_sum += batch_loss * batch_X.shape[0]
            total += batch_X.shape[0]
        logits = np.concatenate(logits_list, axis=0)
        loss = loss_sum / total
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)

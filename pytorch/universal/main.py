import time
from lib.data.loader import DataLoader
from lib.core import create_net
from lib.util.visualizer import Visualizer
from lib.trainer import create_model
from options.train_options import TrainOptions
from options.test_options import TestOptions
import numpy as np


def main():
    opt_train = TrainOptions().parse()
    net = create_net(opt_train)
    data_set_train = DataLoader(opt_train)
    set_train = data_set_train.data_loader()
    data_set_size = len(data_set_train)
    print('The number of training images = %d' % data_set_size)
    opt_test = TestOptions().parse()
    data_set_test = DataLoader(opt_test)
    set_test = data_set_test.data_loader()

    model = create_model(net, opt_train)
    model.setup(opt_train)
    visualizer = Visualizer(opt_train)
    total_iters = 0
    save_iou_data = -1

    for epoch in range(opt_train.epoch_count, opt_train.n_epochs + opt_train.n_epochs_decay):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        loss_sum = {'train_loss': [], 'test_loss': [], 'm_iou': []}
        visualizer.reset()

        model.train()
        for i, data in enumerate(set_train):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += opt_train.batch_size
            epoch_iter += opt_train.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_iters % opt_train.print_freq == 0:
                t_data = iter_start_time - iter_data_time
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt_train.batch_size
                loss_sum['train_loss'].append(losses)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            iter_data_time = time.time()
        data_set_train.reset()

        model.eval()
        for i, data in enumerate(set_test):
            model.set_input(data)
            model.step_verify()
            losses = model.get_current_losses()
            loss_sum['test_loss'].append(losses)
            loss_sum['m_iou'].append(model.m_iou())

        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        losses = {'train_loss': np.mean(np.array(loss_sum['train_loss'])),
                  'test_loss': np.mean(np.array(loss_sum['test_loss'])),
                  'm_iou': np.mean(np.array(loss_sum['m_iou']))
                  }
        visualizer.plot_current_losses(epoch, float(epoch_iter) / data_set_size, losses)

        if epoch % opt_train.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')

        if save_iou_data < losses['m_iou']:
            model.save_networks('test_iou_max')
            save_iou_data = losses['m_iou']

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt_train.n_epochs + opt_train.n_epochs_decay,
                                                              time.time() - epoch_start_time)
              )


if __name__ == '__main__':
    """
    """
    main()


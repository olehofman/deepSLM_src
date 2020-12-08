import os
import timeit
import tracemalloc

import psutil
from sklearn.neural_network._base import softmax, log_loss

from .initialization import create
from .mutation import mutate
from .utils import rmse, accuracy


class DeepSLM:

    def __init__(self, parameters, filename=None, log=False, reporter=True):
        self.parameters = parameters
        self.initialization_sample_size = 1
        self.iterations = parameters['iterations']
        self.mutation_sample_size = parameters['population_size']
        self.log = log
        self.filename = filename
        self.reporter = reporter
        self.parameters['reporter'] = self.reporter

    def fit(self, X, y):

        self.X = X
        self.y = y
        self.iteration = 0

        start_time = timeit.default_timer()
        self._initialization()
        stop_time = "{:.2f}".format(timeit.default_timer() - start_time)
        # collect garbage after each mutation, to make sure maximum amount of RAM is available
        self._collect_RAM_garbage()

        if self.log:
            self._save_run_details(self.best, 0, self.parameters['seed'], stop_time, 'None', 'None')

        for self.iteration in range(1, self.iterations + 1):
            tracemalloc.start()
            start_time = timeit.default_timer()
            cp_time, ncp_time = self._mutation()
            stop_time = "{:.2f}".format(timeit.default_timer() - start_time)
            current, peak = tracemalloc.get_traced_memory()
            print("MEMORY USED: {:.2f}GB; Peak: {:.2f}GB".format(current / 10 ** 9, peak / 10 ** 9))
            tracemalloc.stop()

            if self.log:
                self._save_run_details(self.best, self.iteration + 1, self.parameters['seed'], stop_time, cp_time, ncp_time)

            # process = psutil.Process(os.getpid())
            # # used = float(humanize.naturalsize(process.memory_info().rss).split(' ')[0])
            # print('TOTAL TIME: {}s (NCP: {}s CP: {}s)'.format(stop_time, ncp_time, cp_time))
            # print("MEMORY USED: {:.4f} GB".format(process.memory_info().rss / 1000000000))
            #
            # nbytes = self.best.ncp.nn.coefs[0].nbytes + self.best.ncp.nn.coefs[1].nbytes\
            #          + self.best.ncp.nn.intercepts[0].nbytes + self.best.ncp.nn.intercepts[1].nbytes \
            #          + self.best.ncp.nn.predictions.nbytes + self.best.ncp.nn.input_layer_X.nbytes
            #
            # gbytes = nbytes / 1000000000
            # print("MEMORY NUMPY {:.4f} GB".format(gbytes))


            # collect garbage after each mutation, to make sure maximum amount of RAM is available
            self._collect_RAM_garbage()

    def _initialization(self):
        self.best = None
        for _ in range(self.initialization_sample_size):

            cnn = create(self.parameters)

            cnn_predictions_softmax = softmax(cnn.get_predictions())
            cnn.update_loss(log_loss(self.y, cnn_predictions_softmax))

            if self.initialization_sample_size == 1:
                self.best = cnn
            else:
                self._update_best(cnn)

        self._evaluate_current_best()

        if self.reporter or False:
            # print('\nBest after initialization:')
            self._print_performance(self.best)

    def _generate_run_details(self, cnn, iteration, seed, time, cp_mutation_time, ncp_mutation_time):

        train_rmse = "{:.5f}".format(cnn.train_cnn_rmse)

        train_accuracy = "{:.5f}".format(cnn.train_accuracy)
        test_accuracy = "{:.5f}".format(cnn.test_accuracy)

        train_ce_loss = "{:.5f}".format(cnn.get_loss())
        test_ce_loss = "{:.5f}".format(cnn.test_ce_loss)

        data_row = [str(seed), str(iteration), time, cp_mutation_time, ncp_mutation_time, train_rmse,
                    train_ce_loss, train_accuracy, test_ce_loss, test_accuracy]

        '''data_row = [str(seed), str(iteration), time, cp_mutation_time, ncp_mutation_time, train_rmse, train_ce_loss,
                    train_accuracy]'''

        return data_row

    def _save_run_details(self, cnn, iteration, seed, time, cp_mutation_time, ncp_mutation_time):

        if cp_mutation_time is None:
            cp_mutation_time = 'None'

        run_details = self._generate_run_details(cnn, iteration, seed, time, cp_mutation_time, ncp_mutation_time)

        file = open(self.filename, 'a+')

        file.write(','.join(run_details) + '\n')

        file.close()

    def _print_performance(self, cnn):
        print('---------------------------------------------')
        print('Iteration:', self.iteration)
        # =======================================================================
        # print('Loss (RMSE)\t%.5f' % (cnn.get_loss()))
        # predictions = cnn.get_predictions()
        # print('Training Accuracy\t%.5f%%' % (accuracy(self.y, predictions) * 100))
        #
        # ce_pred = softmax(predictions.copy())
        # ce_loss = log_loss(self.y, ce_pred)'''
        # =======================================================================
        print('---------------------------------------------')
        print('Train CE Loss:\t\t%.5f' % cnn.get_loss())
        print('Test CE Loss:\t\t%.5f' % cnn.test_ce_loss)

        print('Train Accuracy:\t\t''%.5f%%' % (cnn.train_accuracy))
        print('Test Accuracy:\t\t%.5f%%' % (cnn.test_accuracy))
        print('---------------------------------------------')
        # ===============================================================================
        #         pred = cnn.get_predictions()
        #         neuron_accuracies = []
        #         neuron_rmses = []
        #         for i in range(self.y.shape[1]):
        #             y = self.y[:, i]
        #             y_pred = pred[:, i]
        #             neuron_acc = accuracy(y, y_pred) * 100
        #             neuron_accuracies.append(accuracy(y, y_pred) * 100)
        #             print('\tAccuracy for neuron %d\t\t%.2f%%' % (i + 1, neuron_acc))
        #             #print('\tRMSE for neuron %d\t\t%.8f' % (i + 1, rmse(y, y_pred)))
        #             neuron_rmse = rmse(y, y_pred.clip(0, 1))
        #             neuron_rmses.append(neuron_rmse)
        #             print('\tRMSE for neuron %d\t\t%.8f' % (i + 1, neuron_rmse))
        #
        #         average_neuron_rmse = sum(neuron_rmses) / len(neuron_rmses)
        #         average_neuron_accuracy = sum(neuron_accuracies) / len(neuron_accuracies)
        #
        #         print('Average Neuron Accuracy {:.2f}%'.format(average_neuron_accuracy))
        #         print('Average Neuron RMSE {:.5f}'.format(average_neuron_rmse))
        # ===============================================================================
        # print('NCP input layer size: %d' % (len(cnn.ncp.nn.input_layer)))
        # =======================================================================
        # cnn.average_neuron_rmse = average_neuron_rmse
        # cnn.average_neuron_accuracy = average_neuron_accuracy
        # =======================================================================

    def _update_best(self, cnn):
        if self.best is None or cnn.get_loss() < self.best.get_loss():
            self.best = cnn

    def _mutation(self):

        current_best = self.best
        for _ in range(self.mutation_sample_size):
            self._collect_RAM_garbage()
            cnn, cp_time, ncp_time, = mutate(current_best, self.parameters)
            # cnn.update_loss(rmse(self.y, cnn.get_predictions()))

            cnn_predictions_softmax = softmax(cnn.get_predictions().copy())
            ce_loss = log_loss(self.y, cnn_predictions_softmax)
            cnn.update_loss(ce_loss)

            if self.parameters['one_child_keep_child'] or self.mutation_sample_size == 1:
                self.best = cnn
            else:
                self._update_best(cnn)

        self._evaluate_current_best()

        if self.reporter or True:
            # print('\nBest after mutation in iteration', self.iteration, ':')
            self._print_performance(self.best)

        return cp_time, ncp_time

    def predict(self, X, reporter=False):
        return self.best.predict(X, reporter=reporter)

    def _evaluate_current_best(self):
        """ method which evaluates the current best cnn on some performance metrics"""

        current_best = self.best
        train_predictions = current_best.get_predictions()

        """calculate the rmse, accuracy, ce_loss, mean_neuron_rmse and mean_neuron_acc_"""
        train_cnn_rmse = rmse(self.y, train_predictions.clip(0, 1))
        train_mean_neuron_acc = sum([accuracy(
            self.y[:, i], train_predictions[:, i]) for i in range(self.y.shape[1])]) * 100 / self.y.shape[1]
        train_mean_neuron_rmse = sum([rmse(
            self.y[:, i], train_predictions[:, i].clip(0, 1)) for i in range(self.y.shape[1])]) / self.y.shape[1]
        train_accuracy = accuracy(self.y, train_predictions) * 100

        """assign train performance evaluations to cnn"""
        current_best.train_accuracy = train_accuracy
        current_best.train_cnn_rmse = train_cnn_rmse
        current_best.train_mean_neuron_acc = train_mean_neuron_acc
        current_best.train_mean_neuron_rmse = train_mean_neuron_rmse

        """"call predict on test set and assign performance metrics"""
        if self.reporter:
            print("\tStart predict test set")

        # No predict method here because the test set semantics are updated alright "on the go"
        test_predictions = current_best.ncp.nn.test_predictions.copy()
        test_predictions = softmax(test_predictions)

        test_accuracy = accuracy(self.parameters['y_test'], test_predictions) * 100
        test_ce_loss = log_loss(self.parameters['y_test'], test_predictions)

        #assign test performance evaluations to cnn
        current_best.test_ce_loss = test_ce_loss
        current_best.test_accuracy = test_accuracy

    def _collect_RAM_garbage(self):
        import gc

        start_time = timeit.default_timer()
        collected_items = gc.collect()
        garbage_time = "{:.2f}".format(timeit.default_timer() - start_time)

        if self.reporter:
            print("Garbage Collect Time: {}\nCollected Items: {}".format(garbage_time, collected_items))

        return












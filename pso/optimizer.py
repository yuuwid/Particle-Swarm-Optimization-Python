import os
import time
import copy

from pso.particle import Particle
from pso.base_model import set_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import imageio


class PSO:
    """
    A Particle Swarm Optimization 

    Particle Swarm Optimization (PSO) adalah metode komputasi yang mengoptimalkan masalah dengan mencoba secara iteratif untuk meningkatkan solusi kandidat sehubungan dengan ukuran kualitas yang diberikan. (wikipedia)

    Parameters
    -----------
    c1 : float, default=1.0
        Koefisien Akselerasi 1

    c2 : float, default=2.0
        Koefisien Akselerasi 2

    w : float, default=0.5
        Bobot Inersia Pembelajaran

    n_population: int, default=10
        Jumlah populasi Particle

    max_iter: int, default=10
        Maksimum Iterasi Optimasi

    loss_func: string 'mse'|'mae'
        Perhitungan Nilai Error setiap Iterasi
        mse: Mean Square Error
        mae: Mean Absolute Error
    """

    def __init__(self,
                 c1=1.0,
                 c2=2.0,
                 w=0.5,
                 n_population=10,
                 max_iter=10,
                 loss_func='mse'):
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.max_iter = max_iter
        self.n_population = n_population

        if (loss_func == 'mse'):
            self.loss_func = mean_squared_error
        elif loss_func == 'mae':
            self.loss_func = mean_absolute_error
        else:
            self.loss_func = mean_squared_error

        self._set_default_attr()

    def _set_default_attr(self):
        # Buat Kawanan Particle
        self.swarm = []

        # Best error for group
        self.err_best_g = -1

        # Best position for group
        self.pos_best_g = []

        # Best model for group
        self.model_best_g = None

        # Best Parameter Model
        self.best_params_ = {}

    def fit(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return self

    def fit_model(self,
                  model,
                  n_estimators: tuple = (10, 200),
                  max_depth: tuple = (3, 50),
                  boundaries: dict = {}):
        """
        Memasukan Jenis Model yang akan digunakan

        n_estimators : tuple, default=(10, 200)
            Jumlah Parameter Penilai yang akan terbentuk, defaultnya adalah 10 - 200
                -Random Forest, Paramter Penilai adalah Pohon

        max_depth : int, default=None
            Kedalaman maksimum Parameter Penilai. Jika Tidak Ada, maka node diperluas sampai ke semua leaf atau sampai semua leaf mengandung kurang dari sampel min_samples_split.

        boundaries : dict, default={}
            Parameter-parameter tambahan yang digunakan untuk Model Algoritma yang lain.
        """
        _init_boundaries = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
        }

        _init_boundaries = dict(_init_boundaries, **boundaries)

        self.init_model = model
        self.boundaries = _init_boundaries

        return self

    def _set_model(self):
        self.model = set_model(
            model=self.init_model,
            params=self.boundaries,
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            loss_func=self.loss_func
        )

    # Buat Populasi dalam Swarm (Kawanan)
    def _init_swarn(self):
        for i in range(0, self.n_population):
            p = self.model.generate_params_model()
            particle = Particle(p, i)
            self.swarm.append(particle)

    def optimize(self, verbose=None):
        self._set_model()
        self._init_swarn()

        init = time.time()
        i = 0

        self.particle_swarm = []

        # Start optimization loop
        while i < self.max_iter:
            if verbose:
                print('iter_{}'.format(i))

            temp_p_swarn = []
            # Siklus partikel kawanan dalam gerombolan dan Evaluasi Data
            for j in range(0, self.n_population):
                self.swarm[j].evaluate(self.model)

                if verbose == 2:
                    self.swarm[j].print_loss()
                    self.swarm[j].print_particle()

                # Tentukan apakah partikel saat ini adalah yang terbaik (secara global)
                if ((self.swarm[j].err_particle_i < self.err_best_g) or (self.err_best_g == -1)):
                    self.pos_best_g = copy.deepcopy(
                        self.swarm[j].position_particle_i)
                    self.err_best_g = self.swarm[j].err_particle_i
                    self.model_best_g = copy.deepcopy(
                        self.swarm[j].model_particle)

            # Siklus kawanan + perbarui velocity dan posisi particle
            for j in range(0, self.n_population):
                self.swarm[j].update_velocity(
                    self.pos_best_g, self.c1, self.c2, self.w)
                self.swarm[j].update_position(list(self.boundaries.values()))

                temp_p_swarn.append(copy.deepcopy(
                    self.swarm[j]))

            self.particle_swarm.append(copy.deepcopy(temp_p_swarn))

            if verbose:
                print('best loss: {}'.format(self.err_best_g))

            i += 1

        end = time.time()

        if verbose:
            print('total time: {:.2f}s'.format(end - init))

        self.best_params_ = self.format_output()

        return self.model_best_g

    def _optimal_swarm(self):
        pass

    def format_output(self):
        results = {}

        for i, key in enumerate(self.boundaries.keys()):
            results[key] = self.model.get_types(key)(self.pos_best_g[i])

        return results


import copy
import numpy as np


class Particle:
    def __init__(self, parameters, identifier):
        # Particle identifier
        self.identifier = identifier
        # Particle position
        self.position_particle_i = parameters
        # Particle velocity
        self.velocity_particle_i = []
        # Best position Particle sendiri
        self.pos_best_particle_i = []
        # Best error Particle sendiri
        self.err_best_particle_i = -1
        # Error Particle sendiri
        self.err_particle_i = -1
        # Model Algoritma
        self.model_particle = None
        # Jumlah dimensi permasalahan --> berdasarkan parameter
        self.num_dimensions = len(parameters)

        self._init_velocity()

    def _init_velocity(self):
        # Buat initial velocities secara Random
        for i in range(0, self.num_dimensions):
            v = np.random.rand(1)[0]
            self.velocity_particle_i.append(v)

    def print_particle(self):
        print('particle_{}: {}'.format(self.identifier, self.position_particle_i))

    def print_loss(self):
        print('loss_{}: {}'.format(self.identifier, self.err_particle_i))

    # Evaluasi fitness data saat ini
    def evaluate(self, model):
        self.err_particle_i, self.model_particle = model.fit(
            self.position_particle_i)

        # Cek untuk melihat apakah posisi saat ini adalah individu terbaik
        if (self.err_particle_i < self.err_best_particle_i) or (self.err_best_particle_i == -1):
            self.pos_best_particle_i = copy.deepcopy(self.position_particle_i)
            self.err_best_particle_i = self.err_particle_i

    # Update new particle velocity
    def update_velocity(self, pos_best_g, c1, c2, w):
        """
        w : Constant bobot inertia
        c1: Koefisien Akselerasi 1
        c2: Koefisien Akselerasi 2
        """

        for i in range(0, self.num_dimensions):
            r1 = np.random.rand(1)[0]
            r2 = np.random.rand(1)[0]

            cognitive_vel = c1 * r1 * \
                (self.pos_best_particle_i[i] - self.position_particle_i[i])
            social_vel = c2 * r2 * \
                (pos_best_g[i] - self.position_particle_i[i])

            self.velocity_particle_i[i] = w * self.velocity_particle_i[i] + \
                cognitive_vel + social_vel

    # Perbarui posisi partikel berdasarkan pembaruan velocity baru
    def update_position(self, boundaries):
        for i in range(0, self.num_dimensions):
            self.position_particle_i[i] = self.position_particle_i[i] + \
                self.velocity_particle_i[i]

            # Sesuaikan posisi maksimum jika perlu
            if self.position_particle_i[i] > boundaries[i][1]:
                self.position_particle_i[i] = boundaries[i][1]

            # Sesuaikan posisi minimum jika perlu
            if self.position_particle_i[i] < boundaries[i][0]:
                self.position_particle_i[i] = boundaries[i][0]

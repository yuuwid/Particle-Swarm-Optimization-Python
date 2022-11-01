import os
import imageio
import matplotlib.pyplot as plt
import base64


def plotting(opt):
    particle_swarm = opt.particle_swarm
    filenames = []
    i = 0

    best_swarm = []
    index_swarm = 0
    for swarn in particle_swarm:
        avg_err = 0
        total_err = 0

        for par in swarn:
            total_err += par.err_particle_i

        avg_err = total_err / len(swarn)
        best_swarm.append([swarn, avg_err, par.err_best_particle_i])

        index_swarm += 1

    best_swarm = sorted(best_swarm, key=lambda x: (x[2], -x[1]))
    best_swarm = [x[0] for x in best_swarm]
    best_swarm.reverse()

    for swarn in best_swarm:
        x_plot = []
        y_plot = []
        x_best_plot = None
        y_best_plot = None
        for particle in swarn:
            position = particle.position_particle_i
            x_plot.append(position[0])
            y_plot.append(position[1])

        plt.title("Particle Swarm Optimization")
        plt.xlim(-20, 250)
        plt.ylim(-20, 100)
        plt.scatter(x_plot, y_plot, c='b')

        plt.scatter(opt.pos_best_g[0], opt.pos_best_g[1], c='r')

        plt.legend(["Particle", "G Best Particle"])
        plt.xlabel("Error: " + str(round(particle.err_best_particle_i, 2)
                                   ) + "; Error Best: " + str(round(opt.err_best_g, 2)))
        fName = "img/" + str(i) + ".png"
        plt.savefig(fName)
        filenames.append(fName)
        plt.close()

        i += 1

    images = []
    for filename in filenames:
        image = imageio.imread(filename)
        images.append(image)

    imageio.mimwrite('pso-scatter.gif', images, loop=1, duration=0.2)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)

# Jupyter Notebook
def b64():
    b64 = base64.b64encode(
        open("pso-scatter.gif", 'rb').read()).decode('ascii')
    return b64

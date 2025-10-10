import random

def objective_function(x):
    return x ** 2

class Particle:
    def __init__(self, bounds):
        self.position = random.uniform(bounds[0], bounds[1])
        self.velocity = random.uniform(-1, 1)
        self.best_position = self.position
        self.best_score = objective_function(self.position)

    def update_velocity(self, global_best_position, w=0.5, c1=1, c2=2):
        r1 = random.random()
        r2 = random.random()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        if self.position < bounds[0]:
            self.position = bounds[0]
        elif self.position > bounds[1]:
            self.position = bounds[1]

def pso(objective_function, bounds, num_particles, max_iter):
    swarm = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = swarm[0].position
    global_best_score = objective_function(global_best_position)

    for _ in range(max_iter):
        for particle in swarm:
            score = objective_function(particle.position)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position

        for particle in swarm:
            particle.update_velocity(global_best_position)
            particle.update_position(bounds)

    return global_best_position, global_best_score

bounds = [-10, 10]
num_particles = 30
max_iter = 50

best_pos, best_score = pso(objective_function, bounds, num_particles, max_iter)
print(f"Best position: {best_pos}")
print(f"Best score: {best_score}")

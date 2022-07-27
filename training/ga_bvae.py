"""Use a genetic algorithm to find candidate detectors that satisfy AUROC
constraints.

There are three genes:
1. Input Dimensions
2. # Input Channels
3. Input Interpolation Method

Because input dimension has an immense search space, it is divided into 3
buckets (small, medium, and large sizes) and possible solutions are discovered
in each bucket
"""


from typing import Callable, Dict, List
import json
import os
import random
import re


import numpy
import pandas
import PIL
import torch


from bvae import BetaVae
from calibration import calibrate
from find_optimal_decay import optimize_decay
from test import BetaVAEDetector


# TODO: Move these to CL args
N_LATENT = 36
BETA = 2.32
BATCH = 64
TRAIN_DATASET = '../data/train_bvae'
CALIB_DATASET = '../data/calibration_bvae'
TEST_DATASET = '../data/test_bvae'


# Always reseed the RNG on start
random.seed(0)
numpy.random.seed(0)


class GeneticOptimizer:
    """Optimize an objective function using a genetic algorithm.
    
    Args:
        genes: dictionary whose keys are the names of each gene and whose
            values are lists containing all the possible alleles for that
            gene.
        population: the population size.
        objective: objective function, this should be a function that takes a
            dicitionary whose keys are the genes and whose values are the
            corresponding allele value for an individual in the population.
            This function is evaluated to find the fitness of an individual.
        mutation_rate: probability of a mutation occurring.
        logfile: path to write logs (in case GA is interrupted)
    """

    def __init__(self,
                 genes: Dict[str, List[int]],
                 population: int,
                 objective: Callable[[Dict[str, int]], float],
                 mutation_rate: float,
                 logfile: str) -> None:
        self.genes = genes
        self.n_pop = population
        self.objective = objective
        self.mutation_rate = mutation_rate
        self.logfile = logfile
        self.chronicle = []

    def initialize_pop(self) -> numpy.ndarray:
        """Create the initial population.
        
        Returns:
            p x (g + 1) matrix where p is the population size and g is the
            number of genes.  The final column is reserved for reporting the
            fitness values.
        """
        population = numpy.ndarray((self.n_pop, len(self.genes) + 1))
        for individual in range(self.n_pop):
            for gene_idx, alleles in enumerate(self.genes.values()):
                population[individual, gene_idx] = random.choice(alleles)
        return population

    def get_parent_idx(self, cutoffs: List[float]) -> int:
        """Randomly choose a parent for crossover.  The probability of
        choosing an individual as a parent is governed by the cutoffs list.
        
        Args:
            cutoffs - if a random number between 0 and 1 is less than the ith
                element of the cutoff list and greater than the (i-1)th, then
                the ith individual in the current population is chosen to be a
                parent.
        
        Returns:
            The index of the selected parent in the population matrix.
        """
        r = random.random()
        for idx, cutoff in enumerate(cutoffs):
            if r <= cutoff:
                return idx
        raise Exception('Index mismatch')

    def get_known_fitness(self, individual: numpy.ndarray) -> float:
        """Check if the fitness of an individual has already been calculated
        and return it if it has.  This speeds up optimization when duplicate
        individuals appear across generations.
        
        Args:
            individual - vector corresponding to the alleles of a given
                individual.

        Returns:
            The previously calculated fitness for the individual.  If the
            fitness has not previously been calculated, -1 is returned.
        """
        for gen in self.chronicle:
            for other in range(self.n_pop):
                if numpy.array_equal(individual, gen[other, :-1]):
                    return gen[other, -1]
        return -1

    def crossover(self, prev_gen: numpy.ndarray) -> numpy.ndarray:
        """Generate a new population by performing crossover on the previous
        one.
        
        Args:
            prev_gen - p x (g + 1) matrix describing the previous generation.
                Each row corresponds to an individual and the first g columns
                correspond to the individual's alleles for each gene.  The
                final column corresponds to the individual's fitness.
                
        Returns:
            A new generation (also a p x (g + 1) matrix).
        """
        print(prev_gen)
        new_gen = numpy.zeros((self.n_pop, len(self.genes) + 1))
        p_chosen = (prev_gen[:, -1] ** 2) / sum(prev_gen[:, -1] ** 2)
        cutoffs = [p_chosen[0]]
        for individual in range(1, self.n_pop):
            cutoffs.append(cutoffs[-1] + p_chosen[individual])
        for individual in range(self.n_pop):
            parent1 = self.get_parent_idx(cutoffs)
            parent2 = self.get_parent_idx(cutoffs)
            crossover_point = random.randint(1, len(self.genes) - 1)
            offspring = numpy.zeros(len(self.genes))
            offspring[:crossover_point] = prev_gen[parent1, :crossover_point]
            offspring[crossover_point:] = prev_gen[parent2, crossover_point:-1]
            new_gen[individual, :-1] = offspring
        return new_gen

    def mutation(self, generation) -> numpy.ndarray:
        """Randomly mutate genes in a population.
        
        Args:
            generation - p x (g + 1) matrix describing the input generation.
                Each row corresponds to an individual and the first g columns
                correspond to the individual's alleles for each gene.  The
                final column corresponds to the individual's fitness.
                
        Returns:
            A generation with some mutated genes (also a p x (g + 1) matrix).
        """
        for individual in range(self.n_pop):
            for gene_idx, alleles in enumerate(self.genes.values()):
                if random.random() <= self.mutation_rate:
                    generation[individual, gene_idx] = random.choice(alleles)
        return generation

    def breed_ubermensch(self) -> numpy.ndarray:
        """Jawohl!"""
        new_gen = self.crossover(self.chronicle[-1])
        new_gen = self.mutation(new_gen)
        return new_gen

    def step(self) -> None:
        """Create a new generation and evaluate its fitness."""
        if len(self.chronicle) == 0:
            curr_gen = self.initialize_pop()
        else:
            curr_gen = self.breed_ubermensch()
        for individual in range(self.n_pop):
            if self.get_known_fitness(curr_gen[individual, :-1]) >= 0:
                curr_gen[individual, -1] = self.get_known_fitness(
                    curr_gen[individual, :-1])
            else:
                indv_dict = {}
                for gene_idx, gene in enumerate(self.genes):
                    indv_dict[gene] = int(curr_gen[individual, gene_idx])
                curr_gen[individual, -1] = self.objective(indv_dict)
        self.chronicle.append(curr_gen)
        log_msg = f'Results for generation #{len(self.chronicle)}\n' \
                  f'---------------------------------------------\n' \
                  f'{curr_gen}\n' \
                  f'---------------------------------------------\n'
        print(log_msg)
        with open(self.logfile, 'a') as log_f:
            log_f.write(log_msg)


def bvae_objective(params: Dict[str, int]) -> float:
    """Evaluate the fitness of an individual assuming the genes tell you how
    to build a BetaVAE OOD detector.
    
    Args:
        params - the parameters needed to build the individual.  Must include
            size, number of channels, and interpolation method.

    Returns:
        The fitness value of a BetaVAE OOD detector made with this particular
        genetic blueprint.
    """
    # Step 1: Train the Network
    individual_name = f'bvae_n{N_LATENT}_b{BETA}_' \
        f'{"bw" if params["channels"] == 1 else ""}_' \
        f'{"x".join(tuple([str(params["size"])] * 2))}_' \
        f'int{params["interpolation"]}'
    if not os.path.isfile(f'{individual_name}.pt'):
        network = BetaVae(
            n_latent=N_LATENT,
            beta=BETA,
            n_chan=params['channels'],
            input_d=tuple([params['size']] * 2),
            batch=BATCH,
            activation=torch.nn.ReLU(),
            head2logvar='var',
            interpolation=params['interpolation'])
        network.train_self(
            data_path=TRAIN_DATASET,
            epochs=100,
            weights_file=f'{individual_name}.pt')

    # Step 2: Calibrate
    if not os.path.isfile(f'alpha_cal_{individual_name}.json'):
        calibrate(f'{individual_name}.pt', CALIB_DATASET)

    # Step 3: Test
    runner = BetaVAEDetector(
        f'{individual_name}.pt',
        f'alpha_cal_{individual_name}.json',
        window=20,
        decay=10)
    if not os.path.isdir(individual_name):
        os.mkdir(individual_name)
    for vid in os.listdir(TEST_DATASET):
        video_p = vid.replace('.avi', '')
        output_file = os.path.join(
            individual_name,
            f'{individual_name}_{video_p}.xlsx')
        if not os.path.isfile(output_file):
            print(f'Processing {vid}')
            runner.run_detect(os.path.join(TEST_DATASET, vid))
            with pandas.ExcelWriter(output_file) as writer:
                runner.timing_df.to_excel(writer, sheet_name='Times')
                for partition, data_frame in runner.detect_dfs.items():
                    data_frame.to_excel(
                        writer,
                        sheet_name=f'Partition={partition}')

    # Step 4: Find best possible AUROC
    partition_data = {'rain': [], 'brightness': []}
    opts = {}
    cal_data = {}
    with open(f'alpha_cal_{individual_name}.json', 'r') as cal_f:
        cal_data = json.loads(cal_f.read())
    if 'decay' not in cal_data['PARAMS']:
        for part in ['rain', 'brightness']:
            for file in os.listdir(individual_name):
                if part == 'rain' and ('rain0.004' in file or 'empty3.xlsx' in file):
                    with pandas.ExcelFile(os.path.join(individual_name, file)) as xls_f:
                        sheets = pandas.read_excel(xls_f, None)
                        for sheet in sheets:
                            if sheet == 'Partition=rain':
                                partition_data[part].append(sheets[sheet])
                if part == 'brightness' and ('brightness-0.75' in file or 'empty3.xlsx' in file):
                    with pandas.ExcelFile(os.path.join(individual_name, file)) as xls_f:
                        sheets = pandas.read_excel(xls_f, None)
                        for sheet in sheets:
                            if sheet == 'Partition=brightness':
                                partition_data[part].append(sheets[sheet])
            opt_decay, opt_auroc = optimize_decay(partition_data[part])
            opts[part] = opt_auroc
            print(
                f'Optimal decay term for partition {part} is {opt_decay} '
                f'with an AUROC of {opt_auroc}.')
            cal_data = {}
            with open(f'alpha_cal_{individual_name}.json', 'r') as cal_f:
                cal_data = json.loads(cal_f.read())
            if 'decay' not in cal_data['PARAMS']:
                cal_data['PARAMS']['decay'] = {}
            cal_data['PARAMS']['decay'][part] = opt_decay
            with open(f'alpha_cal_{individual_name}.json', 'w') as cal_f:
                cal_f.write(json.dumps(cal_data))
    else:
        opts['rain'] = cal_data['PARAMS']['decay']['rain']
        opts['brightness'] = cal_data['PARAMS']['decay']['brightness']

    # Use F-score to generate fitness from both rain and brightness
    return (2 * opts['rain'] * opts['brightness']) / (opts['rain'] + opts['brightness'])

if __name__ == '__main__':
    # Note: For interpolations modes as of torchvision 0.12, the following apply:
    # 0: NEAREST
    # 1: LANCZOS
    # 2: BILINEAR
    # 3: BICUBIC
    # 4: BOX
    # 5: HAMMING
    small_optimizer = GeneticOptimizer(
        genes={
            'size': [x for x in range(3, 77)],
            'channels': [1, 3],
            'interpolation': [0, 2, 3]},
        population=5,
        objective=bvae_objective,
        mutation_rate=0.2,
        logfile='small.log')
    medium_optimizer = GeneticOptimizer(
        genes={
            'size': [x for x in range(77, 151)],
            'channels': [1, 3],
            'interpolation': [0, 2, 3]},
        population = 5,
        objective=bvae_objective,
        mutation_rate=0.2,
        logfile='medium.log')
    large_optimizer = GeneticOptimizer(
        genes={
            'size': [x for x in range(151, 225)],
            'channels': [1, 3],
            'interpolation': [0, 2, 3]},
        population = 5,
        objective=bvae_objective,
        mutation_rate=0.2,
        logfile='big.log')
    for i in range(100):
        small_optimizer.step()
        medium_optimizer.step()
        large_optimizer.step()

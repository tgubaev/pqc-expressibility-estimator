'''
Defines the ExpressibilityEstimator class.
'''
import os
import pickle

import numpy as np

from qiskit.quantum_info import state_fidelity
from qiskit_aer import AerSimulator


def sample_fidelities(
        pqc,
        num_fidelities,
        filename_prefix=None,
        batch_size=5000,
        seed=None
    ):
    '''
    Samples fidelities using a provided parameterized quantum circuit.

    Parameters
    ----------
    pqc : qiskit.QuantumCircuit
        Parameterized quantum circuit.
    num_fidelities : int
        Number of fidelities to sample.
    filename_prefix : string, optional
        If not None, continuously writes sampled fidelities to
        '<filename_prefix>_fd.dat' and the corresponding meta data to
        '<filename_prefix>_md.pkl'. The default is None.
    batch_size : int, optional
        Number of fidelities sampled in one iteration. The default is
        5000.
    seed : int, optional
        Seed for the random number generator. The default is None.

    Raises
    ------
    FileExistsError
        Raises FileExistsError exception if the file containing fidelities
        already exists.

    Returns
    -------
    fidelities : numpy.array
        NumPy array containing fidelities.

    '''
    # create an array of fidelities
    if filename_prefix is not None:
        # check if the file exists
        fidelities_filename = f'{filename_prefix}_fd.dat'
        if os.path.isfile(fidelities_filename):
            raise FileExistsError(
                f"The file '{fidelities_filename}' already exists."
            )
            
        # create a memory-map to the array of fidelities
        fidelities = np.memmap(
            fidelities_filename,
            dtype=np.float64,
            mode='w+',
            shape=(num_fidelities,)
        )
    else:
        # create an in-memory array of fidelities
        fidelities = np.zeros((num_fidelities,))

    # create a random number generator
    rng = np.random.default_rng(seed=seed)
    
    # process each batch
    num_fidelities_sampled = 0 # the number of already sampled fidelities
    while num_fidelities_sampled < num_fidelities:
        # the number of fidelities in the batch
        num_fidelities_batch = min(
            batch_size, num_fidelities - num_fidelities_sampled
        )

        # sample 2 * num_fidelities_batch parameter sets
        parameter_sets = rng.uniform(
            low=0.0,
            high=2 * np.pi,
            size=(2 * num_fidelities_batch, pqc.num_parameters)
        )

        # create 2 * num_fidelities_batch quantum circuits
        circuits = [
            pqc.assign_parameters(parameter_sets[j, :])
            for j in range(0, 2 * num_fidelities_batch)
        ]

        # create the state-vector simulation backend
        backend = AerSimulator(method='statevector')
        
        # run the quantum circuits and obtain the result
        job = backend.run(circuits)
        result = job.result()

        # compute fidelities in the batch
        fidelities_batch = np.array([
            state_fidelity(
                result.results[2 * j].data.statevector,
                result.results[2 * j + 1].data.statevector
            ) for j in range(0, num_fidelities_batch)
        ])

        # store the fidelities in the array of fidelities
        fidelities[
            num_fidelities_sampled:(
                num_fidelities_sampled + num_fidelities_batch
            )
        ] = fidelities_batch
        
        # delete unused variables
        del parameter_sets
        del circuits
        del backend
        del fidelities_batch
        
        # update the number of already sampled fidelities
        num_fidelities_sampled += num_fidelities_batch
        
        # write to the file if necessary
        if filename_prefix is not None:
            # write fidelities
            fidelities.flush()
            
            # write meta data
            metadata_filename=f'{filename_prefix}_md.pkl'
            with open(metadata_filename, 'wb') as file:
                pickle.dump({
                    'num_qubits': pqc.num_qubits,
                    'num_parameters': pqc.num_parameters,
                    'num_fidelities': num_fidelities,
                    'filename_prefix': filename_prefix,
                    'batch_size': batch_size,
                    'seed': seed,
                    'num_fidelities_batch': num_fidelities_batch,
                    'num_fidelities_sampled': num_fidelities_sampled
                }, file)

    # return the array of fidelities
    return fidelities

def compute_fidelity_pdf(
        fidelities=None,
        filename_prefix=None,
        num_fidelities=None,
        num_bins=75
):
    '''
    Computes fidelity probability density histogram.

    Parameters
    ----------
    fidelities : numpy.array, optional
        NumPy array of fidelities. The default is None.
    filename_prefix : string, optional
        If not None, ignores the provided fidelities and instead reads them
        from '<filename_prefix>_fd.dat'. The default is None.
    num_fidelities : int, optional
        If filename_prefix is not None, num_fidelities fidelities are read
        from the file. If num_fidelities is None, the function will try
        reading it from meta data. The default is None.
    num_bins : int, optional
        Number of bins used to compute the fidelity probability density
        histogram. The default is 75.

    Raises
    ------
    FileNotFoundError
        Raises FileNotFoundError exception if the file containing
        fidelities does not exist.

    Returns
    -------
    hist : numpy.array
        NumPy array of probability densities.
    bin_edges : numpy.array
        NumPy array of bin boundaries.

    '''
    # obtain fidelities
    if filename_prefix is not None:
        # check if the file exists
        fidelities_filename = f'{filename_prefix}_fd.dat'
        if not os.path.isfile(fidelities_filename):
            raise FileNotFoundError(
                f"The file '{fidelities_filename}' does not exist."
            )
        
        # if num_fidelities is not provided, try reading it from meta data
        if num_fidelities is None:
            metadata_filename=f'{filename_prefix}_md.pkl'
            if os.path.isfile(metadata_filename):
                with open(metadata_filename, 'rb') as file:
                    loaded_data = pickle.load(file)
                    num_fidelities = loaded_data['num_fidelities']
            else:
                raise FileNotFoundError(
                    f"num_fidelities is None and the file '{metadata_filename}' does not exist."
                )
        
        # create a memory-map to the array of fidelities
        fidelities = np.memmap(
            fidelities_filename,
            dtype=np.float64,
            mode='w+',
            shape=(num_fidelities,)
        )
    else:
        if fidelities is None:
            raise ValueError(
                "Fidelities were not provided."
            )

    # compute the probability density histogram
    hist, bin_edges = np.histogram(
        fidelities, bins=num_bins, range=(0, 1), density=True
    )

    # return the probability density function
    return hist, bin_edges

def haar_fidelity_pdf(
        num_qubits, F
):
    '''
    Computes Haar fidelity PDF.

    Parameters
    ----------
    F : float
        Fidelity.

    Returns
    -------
    float
        Haar fidelity probability density.

    '''
    num_amplitudes = 2 ** num_qubits # number of amplitudes
    return (num_amplitudes - 1) * (1 - F) ** (num_amplitudes - 2)

def compute_kl_divergence(
        fidelities=None,
        num_qubits=None,
        filename_prefix=None,
        num_fidelities=None,
        num_bins=75,
):
    '''
    Computes KL divergence.

    Parameters
    ----------
    fidelities : numpy.array, optional
        NumPy array of fidelities.
    num_qubits : int, optional
        Number of qubits. If filename_prefix is not None and num_qubits is
        None, num_qubits is read from meta data. 
    filename_prefix : string, optional
        If not None, ignores the provided fidelities and instead reads them
        from '<filename_prefix>_fd.dat'. The default is None.
    num_fidelities : int, optional
        If filename_prefix is not None, num_fidelities fidelities are read
        from the file. If num_fidelities is None, the function will try
        reading it from meta data. The default is None.
    num_bins : int, optional
        Number of bins used to compute the fidelity probability density
        histogram. The default is 75.

    Returns
    -------
    D_KL : float
        KL divergence.

    '''
    # compute the probability density histogram
    hist, bin_edges = compute_fidelity_pdf(
        fidelities=fidelities,
        filename_prefix=filename_prefix,
        num_fidelities=num_fidelities,
        num_bins=num_bins
    )
    
    # obtain num_qubits
    if filename_prefix is not None:
        # if num_qubits is not provided, try reading it from meta data
        if num_qubits is None:
            metadata_filename=f'{filename_prefix}_md.pkl'
            if os.path.isfile(metadata_filename):
                with open(metadata_filename, 'rb') as file:
                    loaded_data = pickle.load(file)
                    num_qubits = loaded_data['num_qubits']
            else:
                raise FileNotFoundError(
                    f"num_qubits is None and the file '{metadata_filename}' does not exist."
                )
    else:
        if num_qubits is None:
            raise ValueError(
                "num_qubits was not provided."
            )
    
    # Initialize the KL-divergence
    D_KL = 0
    
    # Compute the KL-divergence
    for j in range(0, num_bins):
        # Compute the bin midpoint
        bin_midpoint = (bin_edges[j] + bin_edges[j + 1]) / 2
    
        # Compute the bin size
        bin_size = bin_edges[j + 1] - bin_edges[j]
        
        # Integrate the function over the bin
        if hist[j] > 0:
            D_KL = D_KL + hist[j] * np.log(
                hist[j] / haar_fidelity_pdf(num_qubits, bin_midpoint)
            ) * bin_size

    # delete unused variables
    del hist
    del bin_edges
    
    # Return the KL-divergence
    return D_KL

class ExpressibilityEstimator:
    '''
    Implements an expressibility estimator of parameterized quantum circuits.

    Parameters
    ----------
    pqc : qiskit.QuantumCircuit
        Parameterized quantum circuit.
    num_fidelities : int
        Number of fidelities to sample.
    filename_prefix : TYPE
        Prefix of the file containing fidelities.

    '''
    def __init__(self, pqc, num_fidelities, filename_prefix):
        # parameterized quantum circuit
        self.pqc = pqc

        # number of fidelities
        self.num_fidelities = num_fidelities

        # file name prefix
        self.filename_prefix = filename_prefix

        # number of qubits
        self.num_qubits = pqc.num_qubits

        # number of amplitudes
        self.num_amplitudes = 2 ** pqc.num_qubits

        # number of parameters in the PQC
        self.num_parameters = pqc.num_parameters

        # name of the file containing the array of fidelities
        self.fidelities_filename = f'{self.filename_prefix}_fd.dat'
        
        # fidelity data type
        self.fidelities_dtype = np.float64

        # shape of the array of fidelities
        self.fidelities_shape = (num_fidelities,)

    def sample_fidelities(self, batch_size=5000, seed=None):
        '''
        Samples fidelities and stores them in a file with a path
        self.fidelities_filename.

        Parameters
        ----------
        batch_size : int, optional
            Number of fidelities sampled in one iteration. The default is
            5000.
        seed : int, optional
            Seed for the random number generator. The default is None.

        Raises
        ------
        FileExistsError
            Raises FileExistsError exception if the file containing fidelities
            already exists.

        Returns
        -------
        fidelities : numpy.array
            NumPy array containing fidelities.

        '''
        # check if the file exists
        if os.path.isfile(self.fidelities_filename):
            raise FileExistsError(
                f"The file '{self.fidelities_filename}' already exists."
            )

        # Create a random number generator
        rng = np.random.default_rng(seed=seed)
        
        # create a memory-map to the array of fidelities
        fidelities = np.memmap(
            self.fidelities_filename,
            dtype=self.fidelities_dtype,
            mode='w+',
            shape=self.fidelities_shape
        )
        
        # process each batch
        num_fidelities_sampled = 0 # the number of already sampled fidelities
        while num_fidelities_sampled < self.num_fidelities:
            # the number of fidelities in the batch
            num_fidelities_batch = min(
                batch_size, self.num_fidelities - num_fidelities_sampled
            )

            # sample 2 * num_fidelities_batch parameter sets
            parameter_sets = rng.uniform(
                low=0.0,
                high=2 * np.pi,
                size=(2 * num_fidelities_batch, self.num_parameters)
            )

            # create 2 * num_fidelities_batch quantum circuits
            circuits = [
                self.pqc.assign_parameters(parameter_sets[j, :])
                for j in range(0, 2 * num_fidelities_batch)
            ]

            # create the state-vector simulation backend
            backend = AerSimulator(method='statevector')
            
            # run the quantum circuits and obtain the result
            job = backend.run(circuits)
            result = job.result()

            # compute fidelities in the batch
            fidelities_batch = np.array([
                state_fidelity(
                    result.results[2 * j].data.statevector,
                    result.results[2 * j + 1].data.statevector
                ) for j in range(0, num_fidelities_batch)
            ])

            # store the fidelities in the memory-map
            fidelities[
                num_fidelities_sampled:(
                    num_fidelities_sampled + num_fidelities_batch
                )
            ] = fidelities_batch
            fidelities.flush()

            # delete unused variables
            del parameter_sets
            del circuits
            del backend
            del fidelities_batch

            # update the number of already sampled fidelities
            num_fidelities_sampled += num_fidelities_batch

        # return the memory-map
        return fidelities

    def compute_fidelity_pdf(self, fidelities=None, num_bins=75):
        '''
        Computes fidelity probability density histogram.

        Parameters
        ----------
        fidelities : numpy.array, optional
            NumPy array of fidelities. The default is None.
        num_bins : int, optional
            Number of bins used to compute the fidelity probability density
            histogram. The default is 75.

        Raises
        ------
        FileNotFoundError
            Raises FileNotFoundError exception if the file containing
            fidelities does not exist.

        Returns
        -------
        hist : numpy.array
            NumPy array of probability densities.
        bin_edges : numpy.array
            NumPy array of bin boundaries.

        '''
        if fidelities is not None:
            if fidelities.shape != self.fidelities_shape:
                raise ValueError(
                    f"Expected shape {self.fidelities_shape}, but got {fidelities.shape}"
                )
        else:
            # check if the file exists
            if not os.path.isfile(self.fidelities_filename):
                raise FileNotFoundError(
                    f"The file '{self.fidelities_filename}' does not exist."
                )
                
            # create a memory-map to the array of fidelities
            fidelities = np.memmap(
                self.fidelities_filename,
                dtype=self.fidelities_dtype,
                mode='r',
                shape=self.fidelities_shape
            )

        # compute the probability density histogram
        hist, bin_edges = np.histogram(
            fidelities, bins=num_bins, range=(0, 1), density=True
        )

        # return the probability density function
        return hist, bin_edges

    def compute_kl_divergence(self, fidelities=None, num_bins=75):
        '''
        Computes KL divergence.

        Parameters
        ----------
        fidelities : numpy.array, optional
            NumPy array of fidelities. The default is None.
        num_bins : int, optional
            Number of bins used to compute the fidelity probability density
            histogram. The default is 75.

        Returns
        -------
        D_KL : float
            KL divergence.

        '''
        # compute the probability density histogram
        hist, bin_edges = self.compute_fidelity_pdf(
            fidelities=fidelities, num_bins=num_bins
        )
        
        # Initialize the KL-divergence
        D_KL = 0
        
        # Compute the KL-divergence
        for j in range(0, num_bins):
            # Compute the bin midpoint
            bin_midpoint = (bin_edges[j] + bin_edges[j + 1]) / 2
        
            # Compute the bin size
            bin_size = bin_edges[j + 1] - bin_edges[j]
            
            # Integrate the function over the bin
            if hist[j] > 0:
                D_KL = D_KL + hist[j] * np.log(
                    hist[j] / self._haar_fidelity_pdf(bin_midpoint)
                ) * bin_size

        # delete unused variables
        del hist
        del bin_edges
        
        # Return the KL-divergence
        return D_KL
    
    def _haar_fidelity_pdf(self, F):
        '''
        Computes Haar fidelity PDF.

        Parameters
        ----------
        F : float
            Fidelity.

        Returns
        -------
        float
            Haar fidelity probability density.

        '''
        return (self.num_amplitudes - 1) * (1 - F) ** (self.num_amplitudes - 2)
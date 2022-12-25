import qibo
qibo.set_backend("tensorflow")
from sklearn.datasets import load_digits
import numpy as np
from qibo import hamiltonians, gates, models, K
from qibo.hamiltonians import Hamiltonian
from sklearn.metrics import roc_curve, auc
from qibo.symbols import Z
import tensorflow as tf
from itertools import combinations
import cv2

def QAE_circuit(params, nqubits, ntrash, layers, nparams):
    '''
    Create a Quantum Autoencoder Circuit with given parameters
    '''
    circuit = models.Circuit(nqubits)
    if (ntrash <= nqubits/2):
        for l in range(layers):
            for idx in range(ntrash):
                for q in range(nqubits):
                    #phase rotation
                    circuit.add(gates.RY(q,params[q+idx*nqubits+l*ntrash*nqubits]))
                # CZ between trash qubits
                for i,j in combinations(range(nqubits-ntrash,nqubits),2):
                    circuit.add(gates.CZ(i,j))
                # CZ between trash and non-trash qubits
                for i in range(ntrash):
                    for j in range(i,nqubits-ntrash,ntrash):
                        circuit.add(gates.CZ(nqubits-ntrash+((idx+i)%ntrash),j))
    else :
        for l in range(layers):
            for idx in range(nqubits-ntrash):
                for q in range(nqubits):
                    #phase rotation
                    circuit.add(gates.RY(q,params[q+idx*nqubits+l*(nqubits-ntrash)*nqubits]))
                # CZ between trash qubits
                for i,j in combinations(range(nqubits-ntrash,nqubits),2):
                    circuit.add(gates.CZ(i,j))
                for i in range(nqubits-ntrash):
                    for j in range(nqubits-ntrash+i,nqubits,nqubits-ntrash):
                        circuit.add(gates.CZ((idx+i)%(nqubits-ntrash),j))
    for q in range(ntrash):
        circuit.add(gates.RY(nqubits-ntrash+q, params[nparams-ntrash+q]))

    return circuit

def cost_hamiltonian(nqubits, ntrash):
    '''
    Hamiltonian for evaluating Hamming distance based Cost function
    '''
    m0 = K.to_numpy(hamiltonians.Z(ntrash).matrix)
    m1 = np.eye(2 ** (nqubits - ntrash), dtype=m0.dtype)
    ham = hamiltonians.Hamiltonian(nqubits, np.kron(m1, m0))
    return 0.5 * (ham + ntrash)

def data_encoding(dataset, encoding, idx):
    '''
        --------
        Args :
            dataset : Dataset for one-class classification "Handwritten" or "FMNIST"
            encoding : Data encoding method "Amplitude"(Amplitude encoding) or "FRQI"(FRQI encoding)
            idx : Index of the positive data to be trained/tested for one-class classification
        --------
        Return :
            vector_train : train dataset with the positive data
            vector_test_pos : test dataset with the positive data
            vector_test_neg : test dataset with the negative data
            nqubits : the number of qubits required for the given data encoding
    '''

    vector_train = []
    vector_test_pos = []
    vector_test_neg = []
    neg_list = list(range(10))
    neg_list.remove(idx)

    if dataset == "Handwritten":
        digits = load_digits()
        digit_pos = digits.data[np.where(digits.target == idx)]

        if encoding == "Amplitude":
            nqubits = 6   #number of qubits
            # Data Encoding Amplitude
            for i in range(100):
                vector_train.append(np.array(digit_pos[i])/np.linalg.norm(np.array(digit_pos[i])))
            for i in range(100,170):
                vector_test_pos.append(np.array(digit_pos[i])/np.linalg.norm(np.array(digit_pos[i])))
            for idx_neg in neg_list:
                digit_neg = digits.data[np.where(digits.target == idx_neg)]
                for i in range(70):
                    vector_test_neg.append(np.array(digit_neg[i])/np.linalg.norm(np.array(digit_neg[i])))

        elif encoding == "FRQI":
            nqubits = 7 # number of qubits
            digit_pos = digit_pos/16.0
            # Data Encoding FRQI
            for i in range(100):
                vector = np.concatenate((np.cos(np.pi/2*np.array(digit_pos[i])),np.sin(np.pi/2*np.array(digit_pos[i]))))/8.0
                vector_train.append(vector/np.linalg.norm(np.array(vector)))
            for i in range(100,170):
                vector = np.concatenate((np.cos(np.pi/2*np.array(digit_pos[i])),np.sin(np.pi/2*np.array(digit_pos[i]))))/8.0
                vector_test_pos.append(vector/np.linalg.norm(np.array(vector)))
            for idx_neg in neg_list:
                digit_neg = digits.data[np.where(digits.target == idx_neg)]/16.0
                for i in range(70):
                    vector = np.concatenate((np.cos(np.pi/2*np.array(digit_neg[i])),np.sin(np.pi/2*np.array(digit_neg[i]))))/8.0
                    vector_test_neg.append(vector/np.linalg.norm(np.array(vector)))
        else:
            raise ValueError(
                "Amplitude and FRQI encoding is supported"
            )
    elif dataset == "FMNIST":
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        digit_pos = x_train[np.where(y_train == idx)]

        if encoding == "Amplitude":
            nqubits = 8 # number of qubits
            # Data Encoding Amplitude
            for i in range(100):
                vector = cv2.resize(digit_pos[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                vector_train.append(vector/np.linalg.norm(vector))
            for i in range(100,200):
                vector = cv2.resize(digit_pos[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                vector_test_pos.append(vector/np.linalg.norm(vector))
            for idx_neg in neg_list:
                digit_neg = x_train[np.where(y_train == idx_neg)]
                for i in range(100):
                    vector = cv2.resize(digit_neg[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                    vector_test_neg.append(vector/np.linalg.norm(vector))

        elif encoding == "FRQI":
            nqubits = 9 # number of qubits
            # Data Encoding FRQI
            for i in range(100):
                vector = cv2.resize(digit_pos[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                vector = np.concatenate((np.cos(np.pi/2*np.array(vector)),np.sin(np.pi/2*np.array(vector))))/16.0
                vector_train.append(vector/np.linalg.norm(vector))
            for i in range(100,200):
                vector = cv2.resize(digit_pos[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                vector = np.concatenate((np.cos(np.pi/2*np.array(vector)),np.sin(np.pi/2*np.array(vector))))/16.0
                vector_test_pos.append(vector/np.linalg.norm(vector))
            for idx_neg in neg_list:
                digit_neg = x_train[np.where(y_train == idx_neg)]
                for i in range(100):
                    vector = cv2.resize(digit_neg[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                    vector = np.concatenate((np.cos(np.pi/2*np.array(vector)),np.sin(np.pi/2*np.array(vector))))/16.0
                    vector_test_neg.append(vector/np.linalg.norm(vector))
        else:
            raise ValueError(
                "Amplitude and FRQI encoding is supported"
            )
    else:
        raise ValueError(
            "Handwritten digit and Fashion MNIST datasets are supported"
        )

    return vector_train, vector_test_pos, vector_test_neg, nqubits

class VQOCC_circuit():

    def __init__(self, nqubits, ntrash, layers):
        '''
        Variational Quantum One-Class Classifier
        --------
        Args :
            nqubits : The number of qubits
            ntrash : The number of trash qubits
            layers : The number of parameterized quantum circuit layers
        --------
        '''
        assert ntrash < nqubits

        self.nqubits = nqubits
        self.ntrash = ntrash
        self.layers = layers

        if (ntrash <= nqubits/2):
            nparams = ntrash * (nqubits * layers + 1)
        else:
            nparams = (nqubits-ntrash)*nqubits*layers + ntrash

        self.nparams = nparams
        self.params = tf.Variable(tf.random.uniform((nparams,), dtype=tf.float64))
        self.circuit = QAE_circuit(self.params, nqubits, ntrash, layers, nparams)

    def circuit_train_vqocc(self,vector_train,lr=0.1,nepochs=150,batch_size=10,verbose_loss=False):
        '''
        --------
        Args :
            vector_train : train dataset with the positive data
            lr : Learning rate
            nepochs : The number of training epochs
            batch_size : The size of batch for Training
            verbose_loss : returning the loss history
        --------
        '''
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        params = self.params
        loss_history = []
        ntrash = self.ntrash

        circuit = self.circuit

        ham = cost_hamiltonian(self.nqubits,ntrash)
        for ep in range(nepochs):
            # Training Quantum circuit with loss functions evaluated from Hamiltonian
            # using Tensorflow automatic differentiation
            with tf.GradientTape() as tape:
                circuit.set_parameters(params)
                batch_index = np.random.randint(0, len(vector_train), (batch_size,))
                vector_batch = [vector_train[i] for i in batch_index]
                loss = 0
                for i in range(batch_size):
                    final_state = circuit.execute(tf.constant(vector_batch[i]))
                    loss += ham.expectation(final_state)/(ntrash*batch_size)
            grads = tape.gradient(loss, params)
            optimizer.apply_gradients(zip([grads], [params]))
            loss_history.append(loss)

        self.params = params

        if verbose_loss == True :
            return loss_history


    def auc_test(self,vector_test_pos,vector_test_neg):
        '''
        Return :
            auc_measure : AUC measure of the test dataset
        '''
        circuit = self.circuit
        circuit.set_parameters(self.params)
        cost_pos = []
        cost_neg = []
        #Evaluating cost functions for one-class classification
        ntrash = self.ntrash
        ham = cost_hamiltonian(self.nqubits,ntrash)
        for i in range(len(vector_test_pos)):
            final_state = circuit.execute(tf.constant(vector_test_pos[i]))
            cost_pos.append((ham.expectation(final_state)/ntrash).numpy())
        for i in range(len(vector_test_neg)):
            final_state = circuit.execute(tf.constant(vector_test_neg[i]))
            cost_neg.append((ham.expectation(final_state)/ntrash).numpy())

        #Evaluating AUC measure
        y_true = np.array([0]*len(cost_pos)+[1]*len(cost_neg))
        y_score = np.array(cost_pos + cost_neg)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_measure = auc(fpr,tpr)
        return auc_measure

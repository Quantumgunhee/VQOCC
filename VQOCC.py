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

def VQOCC(dataset, encoding, idx, layers, ntrash, lr = 0.1, nepochs = 150, batch_size = 10):
    '''
    Variational Quantum One-Class Classifier returning AUC measure of the test dataset
    --------
    Args :
        dataset : Dataset for one-class classification "Handwritten" or "FMNIST"
        encoding : Data encoding method "Amplitude"(Amplitude encoding) or "FRQI"(FRQI encoding)
        idx : Index of the dataset to be trained/tested for one-class classification
        layers : The number of parameterized quantum circuit layers
        ntrash : The number of trash qubits
        lr : Learning rate
        nepochs : The number of training epochs
        batch_size : The size of batch for Training
    --------
    Return :
        auc_measure : AUC measure of the test dataset
    '''

    vector_target_train = []
    vector_target_test = []
    vector_nontarget = []
    idx_list = list(range(10))
    idx_list.remove(idx)

    if dataset == "Handwritten":
        digits = load_digits()
        target_digit = digits.data[np.where(digits.target == idx)]

        if encoding == "Amplitude":
            nqubits = 6   #number of qubits
            # Data Encoding Amplitude
            for i in range(100):
                vector_target_train.append(np.array(target_digit[i])/np.linalg.norm(np.array(target_digit[i])))
            for i in range(100,170):
                vector_target_test.append(np.array(target_digit[i])/np.linalg.norm(np.array(target_digit[i])))
            for idx_nontarget in idx_list:
                nontarget_digit = digits.data[np.where(digits.target == idx_nontarget)]
                for i in range(70):
                    vector_nontarget.append(np.array(nontarget_digit[i])/np.linalg.norm(np.array(nontarget_digit[i])))

        elif encoding == "FRQI":
            nqubits = 7 # number of qubits
            target_digit = target_digit/16.0
            # Data Encoding FRQI
            for i in range(100):
                vector = np.concatenate((np.cos(np.pi/2*np.array(target_digit[i])),np.sin(np.pi/2*np.array(target_digit[i]))))/8.0
                vector_target_train.append(vector/np.linalg.norm(np.array(vector)))
            for i in range(100,170):
                vector = np.concatenate((np.cos(np.pi/2*np.array(target_digit[i])),np.sin(np.pi/2*np.array(target_digit[i]))))/8.0
                vector_target_test.append(vector/np.linalg.norm(np.array(vector)))
            for idx_nontarget in idx_list:
                nontarget_digit = digits.data[np.where(digits.target == idx_nontarget)]/16.0
                for i in range(70):
                    vector = np.concatenate((np.cos(np.pi/2*np.array(nontarget_digit[i])),np.sin(np.pi/2*np.array(nontarget_digit[i]))))/8.0
                    vector_nontarget.append(vector/np.linalg.norm(np.array(vector)))
        else:
            raise ValueError(
                "Amplitude and FRQI encoding is supported"
            )
    elif dataset == "FMNIST":
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        target_digit = x_train[np.where(y_train == idx)]

        if encoding == "Amplitude":
            nqubits = 8 # number of qubits
            # Data Encoding Amplitude
            for i in range(100):
                vector = cv2.resize(target_digit[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                vector_target_train.append(vector/np.linalg.norm(vector))
            for i in range(100,200):
                vector = cv2.resize(target_digit[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                vector_target_test.append(vector/np.linalg.norm(vector))
            for idx_nontarget in idx_list:
                nontarget_digit = x_train[np.where(y_train == idx_nontarget)]
                for i in range(100):
                    vector = cv2.resize(nontarget_digit[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                    vector_nontarget.append(vector/np.linalg.norm(vector))

        elif encoding == "FRQI":
            nqubits = 9 # number of qubits
            # Data Encoding FRQI
            for i in range(100):
                vector = cv2.resize(target_digit[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                vector = np.concatenate((np.cos(np.pi/2*np.array(vector)),np.sin(np.pi/2*np.array(vector))))/16.0
                vector_target_train.append(vector/np.linalg.norm(vector))
            for i in range(100,200):
                vector = cv2.resize(target_digit[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                vector = np.concatenate((np.cos(np.pi/2*np.array(vector)),np.sin(np.pi/2*np.array(vector))))/16.0
                vector_target_test.append(vector/np.linalg.norm(vector))
            for idx_nontarget in idx_list:
                nontarget_digit = x_train[np.where(y_train == idx_nontarget)]
                for i in range(100):
                    vector = cv2.resize(nontarget_digit[i],dsize=(16, 16), interpolation=cv2.INTER_CUBIC).flatten()
                    vector = np.concatenate((np.cos(np.pi/2*np.array(vector)),np.sin(np.pi/2*np.array(vector))))/16.0
                    vector_nontarget.append(vector/np.linalg.norm(vector))
        else:
            raise ValueError(
                "Amplitude and FRQI encoding is supported"
            )
    else:
        raise ValueError(
            "Handwritten digit and Fashion MNIST datasets are supported"
        )

    assert ntrash < nqubits

    # number of parameters
    if (ntrash <= nqubits/2):
        nparams = ntrash * (nqubits * layers + 1)
    else:
        nparams = (nqubits-ntrash)*nqubits*layers + ntrash

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def QAE_circuit(params):
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

    params = tf.Variable(tf.random.uniform((nparams,), dtype=tf.float64))
    circuit = QAE_circuit(params)

    ham = cost_hamiltonian(nqubits,ntrash)
    for ep in range(nepochs):
        # Training Quantum circuit with loss functions evaluated from Hamiltonian
        # using Tensorflow automatic differentiation
        with tf.GradientTape() as tape:
            circuit.set_parameters(params)
            batch_index = np.random.randint(0, len(vector_target_train), (batch_size,))
            vector_batch = [vector_target_train[i] for i in batch_index]
            loss = 0
            for i in range(batch_size):
                final_state = circuit.execute(tf.constant(vector_batch[i]))
                loss += ham.expectation(final_state)/(ntrash*batch_size)
        grads = tape.gradient(loss, params)
        optimizer.apply_gradients(zip([grads], [params]))

    circuit.set_parameters(params) # setting quantum circuit with optimized parameters
    cost_test = []
    cost_nontarget = []
    #Evaluating cost functions for one-class classification
    for i in range(len(vector_target_test)):
        final_state = circuit.execute(tf.constant(vector_target_test[i]))
        cost_test.append((ham.expectation(final_state)/ntrash).numpy())
    for i in range(len(vector_nontarget)):
        final_state = circuit.execute(tf.constant(vector_nontarget[i]))
        cost_nontarget.append((ham.expectation(final_state)/ntrash).numpy())

    #Evaluating AUC measure
    y_true = np.array([0]*len(cost_test)+[1]*len(cost_nontarget))
    y_score = np.array(cost_test + cost_nontarget)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_measure = auc(fpr,tpr)

    return auc_measure

#Example usage

if __name__ == '__main__':
    idx = 0
    ntrash = 2
    layers = 2

    print("Training index %d with %d trash qubits and %d layers" %(idx,ntrash,layers))
    auc_measure = VQOCC(dataset="Handwritten",encoding="FRQI",idx=idx,layers=layers,ntrash=ntrash)
    print("AUC measure is %f" %auc_measure)

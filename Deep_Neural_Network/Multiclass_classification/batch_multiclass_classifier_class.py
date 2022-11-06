import numpy as np
import matplotlib.pyplot as plt


class Neural_Network_Multiclass_Classifier():

    def __init__(self, X, y, hid_ly_num, hid_ly_neu, learn_rate, learn_iter):
        self.X = X
        self.y = y
        self.hid_ly_num = hid_ly_num
        self.learn_rate = learn_rate
        self.learn_iter = learn_iter

        if type(hid_ly_neu) == int: # broj neurona u skrivenom sloju moze da bude isti za sve i prosledi se kao int ili moze da bude lista, pa se to ovde sredjuje
            self.hid_ly_neu = [] # ako je int onda u listu stavljam vise kopija prosledjenog broja
            for i in range(hid_ly_num):
                self.hid_ly_neu.append(hid_ly_neu)
        else:
            self.hid_ly_neu = hid_ly_neu

        #---

        # inicijalizacija konstanti koje dolaze od parametara
        self.input_neurons = X.shape[1]
        self.output_neurons = y.shape[1] # ovo je const zbog binarne prirode modela
        self.data_points = X.shape[0]
        self.L = []

        #---

        """ KONSTANTA NA SVIM W I B INICALIZOVANIM JE 0.2 ZA W I 0.3 ZA B """
        # inicijalizacija tezina i biasa, za prve su odma dodati
        self.W = [np.random.uniform(0, 0.2, (self.input_neurons, self.hid_ly_neu[0]))]
        self.b = [np.random.uniform(0, 0.3, (1, self.hid_ly_neu[0]))]

        if self.hid_ly_num != 1:
            for i in range(1, self.hid_ly_num): # ovi u sredini preko ovog loopa koji je pomeren za 1
                self.W.append(np.random.uniform(0, 0.2, (self.hid_ly_neu[i - 1], self.hid_ly_neu[i])))
                self.b.append(np.random.uniform(0, 0.3, (1, self.hid_ly_neu[i])))

        # poslednji opet rucno dodati
        self.W.append(np.random.uniform(0, 0.2, (self.hid_ly_neu[-1], self.output_neurons)))
        self.b.append(np.random.uniform(0, 0.3, (1, self.output_neurons)))

    #======

    def CE_Loss(self, yHat):
        return -1 * np.sum( self.y * np.log(yHat) )

    """ KADA POZIVAS PROSLEDI VEKTOR OBAVEZNO, NE RADI NA MATRICI """
    @staticmethod
    def softmax(matrix):
        mx = np.max(matrix, axis = 1)
        mx = mx.reshape( mx.shape[0], 1 )
        
        exp = np.exp(matrix - mx)
        sm = np.sum(exp, axis = 1)

        return (exp.T / sm).T

    @staticmethod
    def transfer(inp, W, b):
        return np.dot(inp, W) + b

    @staticmethod
    def tanh(value):
        return np.tanh(value)

    @staticmethod
    def tanh_derivative(value):
        return 1 - Neural_Network_Multiclass_Classifier.tanh(value)**2

    #=====

    def feedforward(self, X):
        Z = [Neural_Network_Multiclass_Classifier.transfer(X, self.W[0], self.b[0])] # posebno za prvi sloj jer se koristi X
        A = [Neural_Network_Multiclass_Classifier.tanh(Z[0])]

        if self.hid_ly_num != 1: # ovde sve slojeve sem prvog i poslednjeg transferujem
            for i in range(1, self.hid_ly_num):
                Z.append(Neural_Network_Multiclass_Classifier.transfer(A[i-1], self.W[i], self.b[i]))
                A.append(Neural_Network_Multiclass_Classifier.tanh(Z[i]))

        Z.append(Neural_Network_Multiclass_Classifier.transfer(Z[-1], self.W[-1], self.b[-1])) # poslednji transferujem posebno zbog softmaxa
        A.append(Neural_Network_Multiclass_Classifier.softmax(Z[-1]))

        return Z, A

    # razlika u algoritmu backpropa za multiclass i binary je da nemam dA vec odma racunam dZ zbog izvoda na izlaznom sloju na kojem je lakse da se odma izracuna dZ umesto dA pa dZ
    # pored toga malo sam izmenio unutrasnjost backprop alg
    def backpropagation(self, Z, A): 

        # ovde jje razlika sto inicijalizujem odma dZ, dW i db umesto unutra radi lakseg pracenja
        d_ctr = 0 # ovo tehnicki ne treba da postoji ako dole samo stavim ctr = 1 umesto ctr += 1
        ZA_ctr = self.hid_ly_num
        W_ctr = self.hid_ly_num 
        ctr = 0 
        dZ = [ A[ZA_ctr] - self.y ] # slozi se lepo preko formule da dZ izlaznog sloja bude samo yHat - y
        ZA_ctr -= 1

        # postavljajne 0. elementa za dW i db preko formule za backprop
        dW = [ np.dot(A[ZA_ctr].T, dZ[d_ctr]) / self.data_points ]
        db = [ np.sum(dZ[d_ctr]) / self.data_points ]
        ctr += 1 # uradjen prvi ciklus

        # razlika od binary backprop algoritma je u tome sto sam sredinu izdvojio od kraja i pocetka pa idem jos i jedan korak manje
        while ctr != self.hid_ly_num:
            dZ.append( np.dot( dZ[d_ctr], self.W[W_ctr].T) * Neural_Network_Multiclass_Classifier.tanh_derivative(Z[ZA_ctr]) ) # pre znaka * se vidi sta je zapravo trebalo biti dA
            d_ctr += 1 # dodao sam jedan dZ pa povecavam ovde
            ZA_ctr -= 1 # ove smanjujem isto kao u binary backprop
            W_ctr -= 1

            # updateovanje
            dW.append( np.dot(A[ZA_ctr].T, dZ[d_ctr]) / self.data_points )
            db.append( np.sum(dZ[d_ctr]) )
            ctr += 1 # kraj ciklusa

        # napolju radim prvi sloj
        dZ.append( np.dot(dZ[d_ctr], self.W[W_ctr].T) * Neural_Network_Multiclass_Classifier.tanh_derivative(Z[0]) )
        d_ctr += 1

        dW.append( np.dot(self.X.T, dZ[d_ctr]) / self.data_points )
        db.append( np.sum(dZ[-1]) / self.data_points )

        # novi W i b
        i = 0
        j = -1
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - self.learn_rate * dW[j]
            self.b[i] = self.b[i] - self.learn_rate * db[j]
            j -= 1

    #=====

    # funkcija koju pozivas kada zelis da treniras model
    def train(self):

        i = 0

        while i < self.learn_iter: # while loop za odradjivanje iteracija
            if i % 1000 == 0:
                print("Hiljada {} od {}".format( int(i/1000), int(self.learn_iter/1000)))
            Z, A = self.feedforward(self.X) # feed forward deo
            self.L.append( self.CE_Loss(A[-1]) ) # dodajem loss sloja na mesto gde cuvam loss
            self.backpropagation(Z, A) # updateovanje tezina i biasa
            i += 1
        
        plt.plot(self.L)
        plt.show()

    def predict(self, inp):
        _, A = self.feedforward(inp)
        print('Klasa: {}\nSiguran: {}%'.format(np.where(A[-1] == np.max(A[-1]))[0][0], np.max(A[-1]) * 100 )) # ove dve [] su zbog zapisa np.where()

    def accuracy(self, test_data, test_labels):
        test_data_points = test_data.shape[0]
        _, A = self.feedforward(test_data)
        pred = []
        label = []
        for i in range(test_data_points):
            yHat = A[-1][i]
            pred.append( np.where(yHat == np.max(yHat))[0][0] ) # opet ova dva zbog np.where
            
            l = test_labels[i]
            label.append( np.where(l == np.max(l))[0][0] )

        correct = 0
        for i in range(test_data_points):
            if pred[i] == label[i]:
                correct += 1
        
        return correct / test_data_points * 100
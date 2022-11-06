import numpy as np
import random as rnd
import matplotlib.pyplot as plt

"""
Klasa se poziva davanjem dataseta i labela za dataset numpy tensora,
broj skrivenih slojeva inta,
broj neurona u skrivenim slojevima, lista ako hoces razlicite brojeve neurona ili int ako hoces isti za sve slojeve,
stopa ucenja float (0, 1),
broj epoha ucenja int 
velicina minibatcha int
"""
class Neural_Network_Multiclass_Classifier():

    def __init__(self, X, y, hid_ly_num, hid_ly_neu, learn_rate, epochs, batch_size):
        # pravljenje kontrolnog i trening skupa (80:20 podela)
        idxs = list(range(X.shape[0])) # svi indeksi

        tidxs = rnd.sample(idxs, int(X.shape[0] * 0.8)) # indeksi podataka za treniranje
        self.tX = X[tidxs] # skup podataka za treniranje

        cidxs = np.delete(idxs, tidxs) # indeksi kontrolnog skupa
        self.cX = X[cidxs] # kontrolni skup
        #====

        # pravljenje labela za kontrolni i trening set
        self.ty = y[tidxs]
        self.cy = y[cidxs]

        #====
        self.hid_ly_num = hid_ly_num
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # TODO: ovo ubrzaj
        if type(hid_ly_neu) == int: # broj neurona u skrivenom sloju moze da bude isti za sve i prosledi se kao int ili moze da bude lista, pa se to ovde sredjuje
            self.hid_ly_neu = [] # ako je int onda u listu stavljam vise kopija prosledjenog broja
            for i in range(hid_ly_num):
                self.hid_ly_neu.append(hid_ly_neu)
        else:
            self.hid_ly_neu = hid_ly_neu

        #---

        # inicijalizacija konstanti koje dolaze od parametara
        self.input_neurons = X.shape[1]
        self.output_neurons = y.shape[1]
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

    @staticmethod
    def CE_Loss(y, yHat):
        return -1 * np.sum( y * np.log(yHat) )

    @staticmethod
    def softmax(matrix): # stabilni softmax
        mx = np.max(matrix, axis = 1)
        mx = mx.reshape( mx.shape[0], 1 )
        
        exp = np.exp(matrix - mx)
        sm = np.sum(exp, axis = 1)

        return (exp.T / sm).T

    @staticmethod
    def transfer(inp, W, b):
        return np.dot(inp, W) + b

    @staticmethod
    def ReLU(tensor):
        return tensor * (tensor > 0) # u sustini tensor > 0 vraca true/false tensor, a true = 1, false = 0 pa ako elementwise izmnozim dobijem ReLU

    @staticmethod
    def ReLU_derivative(tensor):
        return 1 * (tensor > 0) # ista fora ko gore, samo sto mnozim sa 1 (df/dx = 0, x < 0; df/dx = 1, x > 0 || jer f = 0, x < 0; f = x, x > 0)
        # mogao sam i da stavim int(tensor > 0) mada cenim da je ovo brze jer je int funkcija

    #=====

    def feedforward(self, X):
        Z = [Neural_Network_Multiclass_Classifier.transfer(X, self.W[0], self.b[0])] # posebno za prvi sloj jer se koristi X
        A = [Neural_Network_Multiclass_Classifier.ReLU(Z[0])]

        if self.hid_ly_num != 1: # ovde sve slojeve sem prvog i poslednjeg transferujem
            for i in range(1, self.hid_ly_num):
                Z.append(Neural_Network_Multiclass_Classifier.transfer(A[i-1], self.W[i], self.b[i]))
                A.append(Neural_Network_Multiclass_Classifier.ReLU(Z[i]))

        Z.append(Neural_Network_Multiclass_Classifier.transfer(Z[-1], self.W[-1], self.b[-1])) # poslednji transferujem posebno zbog softmaxa
        A.append(Neural_Network_Multiclass_Classifier.softmax(Z[-1]))

        return Z, A

    # razlika u algoritmu backpropa za multiclass i binary je da nemam dA vec odma racunam dZ zbog izvoda na izlaznom sloju na kojem je lakse da se odma izracuna dZ umesto dA pa dZ
    # pored toga malo sam izmenio unutrasnjost backprop alg
    def backpropagation(self, Z, A, X, y): 

        # ovde jje razlika sto inicijalizujem odma dZ, dW i db umesto unutra radi lakseg pracenja
        d_ctr = 0 # ovo tehnicki ne treba da postoji ako dole samo stavim ctr = 1 umesto ctr += 1
        ZA_ctr = self.hid_ly_num
        W_ctr = self.hid_ly_num 
        ctr = 0 
        dZ = [ A[ZA_ctr] - y ] # slozi se lepo preko formule da dZ izlaznog sloja bude samo yHat - y
        ZA_ctr -= 1

        # postavljajne 0. elementa za dW i db preko formule za backprop
        dW = [ np.dot(A[ZA_ctr].T, dZ[d_ctr]) / self.data_points ]
        db = [ np.sum(dZ[d_ctr]) / self.data_points ]
        ctr += 1 # uradjen prvi ciklus

        # razlika od binary backprop algoritma je u tome sto sam sredinu izdvojio od kraja i pocetka pa idem jos i jedan korak manje
        while ctr != self.hid_ly_num:
            dZ.append( np.dot( dZ[d_ctr], self.W[W_ctr].T) * Neural_Network_Multiclass_Classifier.ReLU_derivative(Z[ZA_ctr]) ) # pre znaka * se vidi sta je zapravo trebalo biti dA
            d_ctr += 1 # dodao sam jedan dZ pa povecavam ovde
            ZA_ctr -= 1 # ove smanjujem isto kao u binary backprop
            W_ctr -= 1

            # updateovanje
            dW.append( np.dot(A[ZA_ctr].T, dZ[d_ctr]) / self.data_points )
            db.append( np.sum(dZ[d_ctr]) )
            ctr += 1 # kraj ciklusa

        # napolju radim prvi sloj
        dZ.append( np.dot(dZ[d_ctr], self.W[W_ctr].T) * Neural_Network_Multiclass_Classifier.ReLU_derivative(Z[0]) )
        d_ctr += 1

        dW.append( np.dot(X.T, dZ[d_ctr]) / self.data_points )
        db.append( np.sum(dZ[-1]) / self.data_points )

        # novi W i b
        i = 0 # jedan od naperd
        j = -1 # jedan sa kraja
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - self.learn_rate * dW[j]
            self.b[i] = self.b[i] - self.learn_rate * db[j]
            j -= 1

    #=====

    # funkcija koju pozivas kada zelis da treniras model
    def train(self):
        batch = self.tX # inicijalizacija potrebnih promenljivih
        batch_l = self.ty

        """ Posto je ovo verzija sa minibatchovanjem, kako radi jeste tako sto uzima ceo dataset i deli ga na male delove koje onda zasebno trenira radi brzine
        sada vise ne postoje iteracije ucenja vec epohe koje se odbrojavaju kada se ceo dataset isprazni batchevima sto samnjuje potrebu za learn iter da bude 
        1 do 2 reda velicine manji tj potrebno je [100, 10000) epoha """
        for i in range(self.epochs):

            if i % (self.epochs/10) == 0: # prikaz epohe na svaku desetinu ukupnih epoha
                print("Epoha {} od {}".format(i, self.epochs))

            while self.batch_size <= batch.shape[0]: # dokle god ima vise datapointa od potrebnih da se napravi minibatch ovo se vrti
                idx = rnd.sample( range(0, batch.shape[0]), self.batch_size ) # indekse po kojima cu da izvlacim iz dataseta pravim ovako, sample() ne pravi ponavljanja a daje random indekse
                mini_batch = batch[idx] # od velikog batcha pravim manji
                mini_batch_l = batch_l[idx] # isto radim i za labele (potrebno zbog backprop)
                
                # radim ucenje
                Z, A = self.feedforward(mini_batch)
                self.backpropagation(Z, A, mini_batch, mini_batch_l)

                batch = np.delete(batch, idx, axis = 0) # brisem delove dataseta koje sam iskoristio
                batch_l = np.delete(batch_l, idx, axis = 0)

            if self.batch_size > batch.shape[0] and batch.shape[0] != 0: # ako batch size nije deljiv sa brojem datapointa onda ulazi u ovaj uslov
                mini_batch = batch # gde ostatak dataseta stavljam u minibatch
                mini_batch_l = batch_l

                # i ucim sa tim minibatchom
                Z, A = self.feedforward(mini_batch)
                self.backpropagation(Z, A, mini_batch, mini_batch_l)

            batch = self.tX # restartujem var
            batch_l = self.ty

            _, cA = self.feedforward(self.cX) # feed forward kontrolnog dela
            self.L.append( self.CE_Loss(self.cy, cA[-1]) ) # dodajem loss sloja na mesto gde cuvam loss

        plt.plot(self.L)
        plt.show()

    def predict(self, inp):
        _, A = self.feedforward(inp)
        print('Klasa: {}.%2f\nSiguran: {}.%2f%'.format(np.where(A[-1] == np.max(A[-1]))[0][0], np.max(A[-1]) * 100 )) # ove dve [] su zbog zapisa np.where()

    def accuracy(self, test_data, test_labels):
        test_data_points = test_data.shape[0]
        _, A = self.feedforward(test_data)
        
        yHat = A[-1]
        t = np.max(yHat, axis = 1)
        t = t.reshape(t.shape[0], 1)

        yHat = (yHat - t).ravel()
        yHat = np.where(yHat == 0)[0]

        #---

        tt = np.max(test_labels, axis = 1)
        tt = tt.reshape(tt.shape[0], 1)

        labels = (test_labels - tt).ravel()
        labels = np.where(labels == 0)[0]

        #---

        res = np.sum(yHat == labels)

        return res / test_data_points * 100
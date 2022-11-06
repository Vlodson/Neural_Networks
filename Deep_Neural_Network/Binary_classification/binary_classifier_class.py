# importovanje potrebnih biblioteka
import numpy as np
from math import e
import random as rnd
import matplotlib.pyplot as plt
#=====

class Neural_Network_Binary_Classifier():

    """ Inicijalizacija klase: treba da se proslede ulazni podaci i labele u obliku np.array gde je .shape[0] broj primera
        zatim treba proslediti broj skrivenih slojeva i broj neurona u tim slojevima
        i na kraju samo broj iteracija i learn rate za backpropagation """
    def __init__(self, X, y, hid_ly_num, hid_ly_neu, learn_rate, learn_iter):

        # inicijalizacija prosledjenih parametara

        # pravljenje kontrolnog i trening skupa (80:20 podela)
        idxs = list(range(X.shape[0])) # svi indeksi

        tidxs = rnd.sample(idxs, int(X.shape[0] * 0.8)) # indeksi podataka za treniranje
        self.tX = X[tidxs] # skup podataka za treniranje

        cidxs = np.delete(idxs, tidxs) # indeksi kontrolnog skupa
        self.cX = X[cidxs] # kontrolni skup
        #====

        # pravljenje labela za kontrolni i trening set
        y = y.reshape((X.shape[0], 1))
        self.ty = y[tidxs]
        self.cy = y[cidxs]

        #====
        self.hid_ly_num = hid_ly_num
        self.learn_rate = learn_rate
        self.learn_iter = learn_iter

        if type(hid_ly_neu) == int: # broj neurona u skrivenom sloju moze da bude isti za sve i prosledi se kao int ili moze da bude lista, pa se to ovde sredjuje
            self.hid_ly_neu = []
            for i in range(hid_ly_num):
                self.hid_ly_neu.append(hid_ly_neu)
        else:
            self.hid_ly_neu = hid_ly_neu

        #=====
        
        # inicijalizacija konstanti koje dolaze od parametara
        self.input_neurons = X.shape[1]
        self.output_neurons = 1 # ovo je const zbog binarne prirode modela
        self.data_points = X.shape[0]
        self.L = []

        #=====
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
    """ Razne funkcije potrebne """
    def MSE(self, y, yHat):
        sq = (y - yHat)**2
        return np.sum(sq)/(2 * self.data_points)

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

    @staticmethod
    def sigmoid(value):
        return 1 / (1 + e**(-1 * value))

    @staticmethod
    def sigmoid_derivative(value):
        return (1 - Neural_Network_Binary_Classifier.sigmoid(value)) * Neural_Network_Binary_Classifier.sigmoid(value)

    #======

    def feedforward(self, X): # nista specijalno, samo transferujem (objasnjena gore metoda transfer) (vrv mi nije trebala, mogao sam i ovde samo ispisati al ajd krucenje malo)
        Z = [Neural_Network_Binary_Classifier.transfer(X, self.W[0], self.b[0])] # posebno za prvi sloj jer se koristi X
        A = [Neural_Network_Binary_Classifier.ReLU(Z[0])]

        if self.hid_ly_num != 1: # ovde sve slojeve sem prvog i poslednjeg transferujem
            for i in range(1, self.hid_ly_num):
                Z.append(Neural_Network_Binary_Classifier.transfer(A[i-1], self.W[i], self.b[i]))
                A.append(Neural_Network_Binary_Classifier.ReLU(Z[i]))

        Z.append(Neural_Network_Binary_Classifier.transfer(Z[-1], self.W[-1], self.b[-1])) # poslednji transferujem posebno zbog sigmoida
        A.append(Neural_Network_Binary_Classifier.sigmoid(Z[-1]))

        return Z, A
    
    #================

    # e sad ovaj je malo ludji becar. backprop sam po sebi sam obj u info.txt algoritamski cu ovde komentare
    def backpropagation(self, Z, A): # f ne vraca nista jer samo treba da updateuje W i b
        
        # posto treba da pratim 4+ stvari istovremeno na tri razlicita mesta i nisam vrsan u 5D sahu i kvantnoj fizici, odlucio sam da napravim brojace za stvari posebno
        d_ctr = 0 # neke sam grupisao kao ovaj koji broji sve gradijente
        ZA_ctr = self.hid_ly_num # ovaj brojac je za Z i A liste iz transfera koje moraju biti tacno jedan ispred W brojaca
        W_ctr = self.hid_ly_num # brojac za tezine koji mora kasniti za jedan
        ctr = 0 # brojac sloja u kojem se nalazim
        dA, dZ, dW, db = [], [], [], [] # liste u koje cu ubaciti gradijente

        while ctr != self.hid_ly_num + 1: # ova grdosija radi i to samo uz boziju pomoc 

            if ctr == 0: # opet odvojeno radim za poslednji sloj jer dA treba da se gleda preko izvoda lossa, a i dZ mora sa izvodom sigme
                dA.append( (A[ZA_ctr] - self.ty)/self.data_points )
                dZ.append( dA[d_ctr] * Neural_Network_Binary_Classifier.sigmoid_derivative(Z[ZA_ctr]) )
                ZA_ctr -= 1 # spustam ZA za jedan jer mi sada treba pretposlednji sloj

                dW.append( np.dot( A[ZA_ctr].T, dZ[d_ctr] )/self.data_points ) # pravim slope za W i b i NE POVECAVAM d_ctr jer sam i dalje na nultom elementu
                db.append( np.sum(dZ[d_ctr])/self.data_points )
                ctr += 1 # ctr +1 jer sam ovde zavrsio jedan prolaz tehnicki

            dA.append( np.dot( dZ[d_ctr], self.W[W_ctr].T ) ) # ovde sada koristim i W brojac koji je trenutno na poslednjim tegovima
            d_ctr += 1 # ovde uvecam d_ctr za 1 jer mi treba dA sledeci za dZ
            dZ.append( dA[d_ctr] * Neural_Network_Binary_Classifier.ReLU_derivative(Z[ZA_ctr]) ) # koristim tanh izvod sada
            ZA_ctr -= 1 # spustam oba da se spreme za sledeci sloj (i slope tezine jer treba A jednog ispod)
            W_ctr -= 1

            dW.append( np.dot( A[ZA_ctr].T, dZ[d_ctr] )/self.data_points ) # sredjujem W i b slope
            db.append( np.sum(dZ[d_ctr])/self.data_points )
            # d_ctr += 1 # sada povecavam d_ctr jer sam napravio jos jedan element u svakom

            if ctr == self.hid_ly_num + 1: # posebno za prvi skriveni sloj i prve W i b jer se koristi X
                dA.append( np.dot( dZ[d_ctr], self.W[W_ctr].T ) )
                dZ.append( dA[d_ctr] * Neural_Network_Binary_Classifier.ReLU_derivative(Z[ZA_ctr]) )
                
                dW.append( np.dot( self.tX.T, dZ[d_ctr] )/self.data_points )
                db.append( np.sum(dZ[d_ctr])/self.data_points )

            ctr += 1 

        #========

        # updateovanje W i b
        i = 0
        j = -1
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - self.learn_rate * dW[j]
            self.b[i] = self.b[i] - self.learn_rate * db[j]
            j -= 1

        #=========

    # funkcija koju pozivas kada zelis da treniras model
    def train(self):

        i = 0

        while i < self.learn_iter: # while loop za odradjivanje iteracija 
            Z, A = self.feedforward(self.tX) # feed forward deo
            _, cA = self.feedforward(self.cX) # feed forward kontrolnog dela
            self.L.append( self.MSE(self.cy, cA[-1]) ) # dodajem loss sloja na mesto gde cuvam loss
            self.backpropagation(Z, A) # updateovanje tezina i biasa
            i += 1
        
        plt.plot(self.L)
        plt.show()


    # funkcija za klasifikaciju/predikciju, samo ubacis list/element koji treba da se predvidi
    # ovde ulazi pojedinacni primeri koji trebaju da se klasifikuju
    # moze se ovo ulepsati tako sto dodam neki tekst kod printovanja ili nesto ali ne treba za sada
    def predict(self, inp):
        _, A = self.feedforward(inp)
        print(A[-1][0][0]) # zasto 3 []? -> 1. A je lista vektora 2. koji vektor (iako je samo jedan) 3. posto je vektor, moram da pokazem koja coord vektora (iako samo 1)
        

    # testiranje tacnosti mreze, sve iznad 0.5 ce pripadati jednoj, sve ispod 0.5 drugoj klasi, pored toga ako je DA/NE mreza u pitanju, iznad 0.5 je da ispod je ne
    # sve mora biti vektorizovano
    def accuracy(self, test_data, test_labels):
        test_data_points = test_data.shape[0]
        _, A = self.feedforward(test_data)
        pred = []
        for yHat in A[-1]: # ovde pravim listu predpostavki labele na osnovu ovoga gore napisanog
            if yHat[0] < 0.5: # opet zbog gluposti numpy i vektora ima []
                pred.append(0)
            else:
                pred.append(1)
        
        correct = 0
        for i in range(test_data_points): # samo najobicnija provera tacnosti, ako se poklapaju ctr ++
            if pred[i] == test_labels[i]: # opet [] zbog vektor gluposti
                correct += 1

        print(correct / test_data_points * 100, '%\n')

        
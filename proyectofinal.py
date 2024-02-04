import numpy as np #importa la lib numpy
import matplotlib.pyplot as plt #importando la biblioteca matplotlib.pyplot como plt


def sigmoid(x): #funcion de activacion sigmoidal
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivada(x): #deriva de la funcion de activacion sigmoidal
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x): #funcion de activacion tangente hiperbbolica
    return np.tanh(x)

def tanh_derivada(x): #se deriva de la funcion de activacion tangente hiperbolica
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivada
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_derivada

        # inicializo los pesos
        self.weights = []
        self.deltas = []
        # capas = [2,3,2]
        # rando de pesos varia entre (-1,1)
        # asigno valores aleatorios a capa de entrada y capa oculta
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # asigno aleatorios a capa de salida
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Agrego columna de unos a las entradas X
        # Con esto agregamos la unidad de Bias a la capa de entrada
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # Calculo la diferencia en la capa de salida y el valor obtenido
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]
            
            # Empezamos en el segundo layer hasta el ultimo
            # (Una capa anterior a la de salida)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
            self.deltas.append(deltas)

            # invertir
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiplcar los delta de salida con las activaciones de entrada 
            #    para obtener el gradiente del peso.
            # 2. actualizo el peso restandole un porcentaje del gradiente
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print('epochs:', k)

    def predict(self, x): #este metodo hace predicciones con la red neuronal
        ones = np.atleast_2d(np.ones(x.shape[0])) #asegurarse de que x tenga 2 dimensiones
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)): # recorre todas las capas de la red neuronal
            a = self.activation(np.dot(a, self.weights[l]))
        return a #devuelve la saida que arroja la red

    def print_weights(self): #imprime los pesos de la conex en la red
        print("LISTADO PESOS DE CONEXIONES")
        for i in range(len(self.weights)):
            print(self.weights[i])

    def get_deltas(self): #obtiene los deltas de la red neuronal
        return self.deltas
    
# funcion Coche Evita obstáculos
nn = NeuralNetwork([2,3,2],activation ='tanh')
X = np.array([[0, 0],   # sin obstaculos
              [0, 1],   # sin obstaculos
              [0, -1],  # sin obstaculos
              [0.5, 1], # obstaculo detectado a derecha
              [0.5,-1], # obstaculo a izq
              [1,1],    # demasiado cerca a derecha
              [1,-1]])  # demasiado cerca a izq

y = np.array([[0,1],    # avanzar
              [0,1],    # avanzar
              [0,1],    # avanzar
              [-1,1],   # giro izquierda
              [1,1],    # giro derecha
              [0,-1],   # retroceder
              [0,-1]])  # retroceder
nn.fit(X, y, learning_rate=0.03,epochs=15001)

 #------------------------- 

index=0
for e in X:
    print("X:",e,"y:",y[index],"Network:",nn.predict(e))
    index=index+1
    
 
    
deltas = nn.get_deltas()
valores=[]
index=0
for arreglo in deltas:
    valores.append(arreglo[1][0] + arreglo[1][1])
    index=index+1

# crear un gráfico de línea con valores en el eje y y el rango de épocas en el eje x
# len(valores) devuelve la longitud de la lista 'valores', y range(len(valores)) crea el rango de índices para el eje x
# el color de la línea se establece en azul ('b')
plt.plot(range(len(valores)), valores, color='b')

# estableciendo el límite del eje y entre 0 y 1
plt.ylim([0, 1])

#etiquetando el eje y como costo 
plt.ylabel('Cost')

#etiqueta el eje x como 'Epochs' epocas
plt.xlabel('Epochs')

#ajusta automáticamente el diseño para que no haya superposición de elementos
plt.tight_layout()

#se muestra el grafico en una ventana adicional
plt.show()

import nltk

import discord

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import json
import random
import pickle

llave= "aquivallavedeldiscord"  #discord llave

#abrimos el archivo con el contenido de las etiqutas,patrones y respuestas
with open("contenido.json",encoding='utf-8') as archivo:
    datos=json.load(archivo)   #guardamos los datos del archivo en la variable datos
try:
    with open("variables.pickle","rb") as archivoPickle:
        palabras,tags,entrenamiento,salida = archivoPickle.load(archivoPickle)
except:

    palabras=[]
    tags=[]

    #este almacena el patron de palabras
    auxX=[]

    #cada entrada de este auxiliar se corresponde al anterior auxiliar
    auxY=[]



#iteramos sobre los datos primero a contenido
    for contenido in datos["contenido"]:
        #luego iteramos sobre los patrones
        for patrones in contenido["patrones"]:

            #mediante el tokenize dividimos la frase en palabras , teniendo en cuenta tambien los signos
            #no solo es un split    que divide por espacios
            auxPalabra= nltk.word_tokenize(patrones) #almacenamos la palabra

            #agregamos todos los tokens de la lista de auxPalabra hacia la variable palabras
            palabras.extend(auxPalabra)

            #guardamos las frases con la lista tokenizada de palabras hacia nuestro auxiliar

            auxX.append(auxPalabra)



            #en nuestro auxiliarY agregamos al final los tags
            auxY.append(contenido["tag"])

            #evitaremos guardar los tags repeteidos ya que estamos en una
            #iteracion por lo que verficamos que no este en la lista antes
            #de agregar
            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])


    # print(palabras)
    # print(auxX)
    # print(auxY)
    # print(tags)

    #con el steemer hacemos mas comprensible cada palabra ,debido a que
    #convertirmos la palabra a sus raices,osea la simplificamos
    palabras= [stemmer.stem(w.lower()) for w in palabras if w!="?" ]



    #ordenamos la lista de palabras y eliminamos duplicados
    palabras = sorted(list(set(palabras)))


    #ordenamos la lista de tags
    tags= sorted(tags)

#aca guardaremos todas las bolsas de palabras que son 0 y 1 para hacer el entrenamiento
    entrenamiento = []
    salida =[]

    salidaVacia= [0 for _ in range(len(tags))]

    #recorremos nuestro auxiliarX el cual tien lista de frases tokenizadas
    #con el enumarate podremos obtener la palabra y guardarlo en la variable documento, y en ex su indice
    for x,documento in enumerate(auxX):
        bolsaPalabras=[]
        auxPalabra=[stemmer.stem(w.lower()) for w in documento]

        for w in palabras:
            if w in auxPalabra:
                bolsaPalabras.append(1)
            else:
                bolsaPalabras.append(0)
        filaSalida= salidaVacia[:]
        filaSalida[tags.index(auxY[x])]=1
        entrenamiento.append(bolsaPalabras)
        salida.append(filaSalida)

    print(entrenamiento)
    print(salida)

    #TF trabaja con matrices por eso usamos numpy para castear
    entrenamiento = numpy.array(entrenamiento)
    salida = numpy.array(salida)
    #hasta aca preparamos los datos para alimentar al modelo

    with open("variables.pickle","wb") as archivoPickle:
        pickle.dump((palabras,tags,entrenamiento,salida),archivoPickle    )

# tensorflow.reset_default_graph()
#hacemos esto para asegurar evitar toda configuracion anterior
tensorflow.compat.v1.reset_default_graph()

#definimos la forma de entrada para nuestro modelo
red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,len(salida[0]),activation="softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red) #tipo de analisis neuronal



modelo.fit(entrenamiento,salida,n_epoch=1000,batch_size=10,show_metric=True)
modelo.save("modelo.tflearn")

def mainBot():

    global llave
    cliente = discord.Client()
    while True:

        @cliente.event
        async def on_message(mensaje):
            if mensaje.author==cliente.user:
                return
            # entrada= input("Tu: ")
            cubeta= [0 for _ in range(len(palabras))]
            entradaProcesada = nltk.word_tokenize(mensaje.content )
            entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
            for palabraIndividual in entradaProcesada:
                for i,palabra in enumerate(palabras):
                    if palabra == palabraIndividual:
                        cubeta[i] = 1
            resultados = modelo.predict([numpy.array(cubeta)])
            print(resultados)

            #tomamos la prediccion con el mayor valor , osea el mas cercano

            resultadosIndices = numpy.argmax(resultados)
            valorObtenido=resultados[0,resultadosIndices]
            print("Resultado mayor", valorObtenido)
            global mensajeRespuesta
            if valorObtenido>0.6:


                tag= tags[resultadosIndices]

                for tagAux in datos["contenido"]:
                    if tagAux["tag"]== tag:
                        respuesta = tagAux["respuestas"]
                        mensajeRespuesta=random.choice(respuesta)

                # print("BOT: ",random.choice(respuesta))
            else:
                mensajeRespuesta="No te entendí, por favor verifica tu respuesta"


            await mensaje.channel.send(mensajeRespuesta)
        cliente.run(llave)

mainBot()


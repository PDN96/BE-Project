import time
import socket
import nxt
import nxt.bluesock
import cv2
import numpy as np

class VideoStreamHandler(object):
    # create neural network
    def sigmoid(self, Z):
        A = 1/(1+np.exp(-Z))
        return A

    def relu(self, Z):
        A = np.maximum(0,Z)
        return A

    def predict(self, samples):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        Z1 = np.dot(self.W1, samples.T) + self.b1[0]
        A1 = self.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2[0]
        A2 = self.sigmoid(Z2)
        index = np.argmax(A2)
        return index
    
    def __init__(self):
        print("Loading data set...")
        image_array = np.zeros((1, 1800), 'float')
        label_array = np.zeros((1, 7), 'float')
        # Retrieve a list of pathname that matches the below expr
        parameters=np.load('newparameters.npz')
        self.W1 = parameters["Weights1"]
        print self.W1.shape
        self.W2 = parameters["Weights2"]
        print self.W2.shape
        self.b1 = parameters["Bias1"]
        print self.b1.shape
        self.b2 = parameters["Bias2"]
        print self.b2.shape
        self.model = cv2.ml.ANN_MLP_load('mlp_xml/mlp_1524216071.xml')
        self.server_socket = socket.socket()
        self.server_socket.bind(('0.0.0.0', 8005))
        self.server_socket.listen(0)
        stream_bytes = ' '
        b = nxt.bluesock.BlueSock('00:16:53:0F:A2:91').connect()
        # stream video frames one by one
        try:
            self.connection = self.server_socket.accept()[0].makefile('rb')
            b.start_program('servoblue.rxe')
            b.message_write(0,str(3))
            b.message_write(8,str(3))
            count=0;
            images = np.zeros((1, 1800),'int')
            while count<20:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    image_array1 = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), 0)
                    image_array = image_array1.reshape(1, 1800).astype(np.uint8)
                    images = np.vstack((images, image_array))
                    count+=1
            mean=np.mean(images)
            std=np.std(images)
            b.message_write(0,str(1.95))
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    image_array2 = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), 0)
                    #image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
                    cv2.imshow('image', image_array2)
                    image_array101 = image_array2.reshape(1, 1800).astype(np.uint8)
                    cap = ((image_array101-mean)/std)
                    p = self.predict(cap)
                    # stop conditions
                    #b.message_write(0,int(3-abs(p-3)))
                    b.message_write(8,str(p))
                    print p
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        b.stop_program()
                        self.connection.close()
                        self.server_socket.close()
                        break
            cv2.destroyAllWindows()
        finally:
            self.connection.close()
            self.server_socket.close()
            b.stop_program()
            print "Connection closed on thread 1"
            
    

if __name__ == '__main__':

    VideoStreamHandler()

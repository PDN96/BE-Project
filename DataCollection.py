import numpy as np
import cv2
import pygame
from pygame.locals import *
import socket
import time
import nxt
import nxt.bluesock


class CollectTrainingData(object):
    def __init__(self):
        self.server_socket = socket.socket()
        self.server_socket.bind(('0.0.0.0', 8005))
        self.server_socket.listen(0)
        self.connection = self.server_socket.accept()[0].makefile('rb')
        # create labels
        self.k = np.identity(7,'int')
        self.collect_image()
        
    def collect_image(self):
        pygame.init()
        saved_frame = 0
        total_frame = 0
        driveval=1;
        steerval=1;
        screen = pygame.display.set_mode([200, 200])
        pygame.display.set_caption("My Game")
        pygame.joystick.init()
        b = nxt.bluesock.BlueSock('00:16:53:0F:A2:91').connect()
        b.start_program('servoblue.rxe')
        b.message_write(0,str(driveval))
        b.message_write(9,str(steerval))
        # collect images for training
        print 'Start collecting images...'
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 1800),'int')
        label_array = np.zeros((1, 7),'int')
        done=False;
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        e3 = cv2.getTickCount()
        # stream video frames one by one
        try:
            stream_bytes = ' '
            e3prev=0;
            while done==False:
                e3 = cv2.getTickCount()
                for event in pygame.event.get(): # User did something
                    if event.type == pygame.QUIT : # If user clicked close
                        done=True
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), 0)
                    #cv2.imshow('image', image)
                    total_frame += 1
                    driveval = round(((joystick.get_axis( 1 ))+1)*3)
                    steerval = int(round(6-(((joystick.get_axis( 4 ))+1)*3)))
                    b.message_write(0,str(driveval))
                    b.message_write(8,str(steerval))
                    if ((e3-e3prev)/cv2.getTickFrequency())>=0.3:
                        saved_frame += 1
                        #cv2.imwrite('training_images/frame{:>05}.jpg'.format((steerval+1)*10000+saved_frame), image)
                        temp_array = image.reshape(1, 1800).astype(np.uint8)
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, self.k[steerval]))
                        e3prev=e3
                        print saved_frame
                

#save training images and labels
            train = image_array[1:, :]
            train_labels = label_array[1:, :]
            print np.size(train)
            print np.size(train_labels)
### save training data as a numpy file
            file_name = str(int(time.time()))
            directory = "training_data" 
            np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
            e2 = cv2.getTickCount()
           # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print 'Streaming duration:', time0
            print(train.shape)
            print(train_labels.shape)
            print 'Total frame:', total_frame
            print 'Saved frame:', saved_frame
            print 'Dropped frame', total_frame - saved_frame
        finally:
            self.connection.close()
            self.server_socket.close()
            b.stop_program()
            print "done"

if __name__ == '__main__':
    CollectTrainingData()
